// src/poly/nemotron_h.cpp - the layers of Nemotron-H with routed experts (nemotron_h.hpp).  The reference for the math
// is llama.cpp's graph at 08c252e22: src/models/nemotron-h.cpp (the layer kinds, the attention, the MoE with its latent
// projections and shared expert), mamba-base.cpp's build_mamba2_layer, llama-graph.cpp's build_moe_ffn, and
// ggml-cpu's ssm_conv and ssm_scan (the kernels: strata/poly/mamba2.hpp, tested by mamba2_test.cu).
#include "nemotron_h.hpp"

#include "fattn.hpp"

#include "runtime.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/poly/mamba2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time
constexpr int64_t kGatingSigmoid = 2;              // llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID

int64_t of_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size()) throw std::runtime_error("a per-layer value of the GGUF ends before its layers");
    return (int64_t) v.items[(size_t) il].u;
}

// Super's files have the norms' epsilon under the LayerNorm's name alone; llama.cpp takes it for the RMS norms too.
const Names kRmsEps{"attention.layer_norm_rms_epsilon", {"attention.layer_norm_epsilon", nullptr}};

}  // namespace

const Names* NemotronH::meta_alias(const char* key) const {
    return std::strcmp(key, kRmsEps.name) == 0 ? &kRmsEps : nullptr;
}

void NemotronH::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_mamba_ = 0;
}

void NemotronH::check_draft_block() {
    throw std::runtime_error("this model's block that guesses ahead (its MTP head) is not run yet; start it with "
                             "--no-draft");
}

void NemotronH::load_geometry() {
    load_shape();
    if (draft_) check_draft_block();   // before the layers: the block would be one of them
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        return *v;
    };
    auto num = [&](const char* key, double otherwise) {
        const MetaValue* v = find(key);
        return v && v->is_num() ? v->num() : otherwise;
    };

    // a layer's kind (llama.cpp's is_recr()): no key-value heads and no FFN width is a Mamba-2 layer, no FFN width
    // an attention layer, the others routed experts
    const MetaValue& kv = must("attention.head_count_kv");
    const MetaValue& ff = must("feed_forward_length");
    kinds_.assign((size_t) n_layer_, Kind::kMamba);
    int counts[3] = {};
    max_kv_ = 0;
    for (int il = 0; il < n_layer_; ++il) {
        const int64_t k = of_layer(kv, il), f = of_layer(ff, il);
        const Kind kind = k == 0 && f == 0 ? Kind::kMamba : f == 0 ? Kind::kAttn : Kind::kMoe;
        kinds_[(size_t) il] = kind;
        ++counts[(int) kind];
        if (kind == Kind::kAttn) {
            need(max_kv_ == 0 || k == max_kv_, "the attention layers have different key-value heads");
            max_kv_ = (int) k;
        }
    }

    // Mamba-2: ssm_in gives z (d_inner) | xBC (d_inner + 2 * groups * d_state) | dt (a value a head)
    d_conv_ = shape_i("ssm.conv_kernel");
    d_inner_ = shape_i("ssm.inner_size");
    d_state_ = shape_i("ssm.state_size");
    n_group_ = shape_i("ssm.group_count");
    n_ssm_head_ = shape_i("ssm.time_step_rank");
    need(d_conv_ == 4, "ssm.conv_kernel is not 4");
    need(n_ssm_head_ > 0 && n_group_ > 0 && d_inner_ % n_ssm_head_ == 0 && n_ssm_head_ % n_group_ == 0 &&
             d_inner_ % n_group_ == 0, "bad ssm sizes");
    need(d_state_ % 32 == 0 && d_state_ <= 256, "ssm.state_size is not 32 to 256 in 32s");
    ssm_head_dim_ = d_inner_ / n_ssm_head_;
    conv_ch_ = d_inner_ + 2 * n_group_ * d_state_;
    d_in_proj_ = 2 * d_inner_ + 2 * n_group_ * d_state_ + n_ssm_head_;

    // attention: no rotation is applied (llama.cpp's build_attention_layer has none)
    n_head_ = shape_i("attention.head_count");
    need(n_head_ > 0 && n_head_ <= 256, "attention.head_count is not one number for every layer");
    head_dim_ = shape_i("attention.key_length");
    need(shape_i("attention.value_length") == head_dim_, "the keys and values have different widths");
    need(head_dim_ % 32 == 0 && head_dim_ <= 256, "bad head size");
    need(counts[(int) Kind::kAttn] == 0 || (max_kv_ > 0 && n_head_ % max_kv_ == 0), "bad attention.head_count(_kv)");
    need(num("attention.scale", 0.0) == 0.0, "attention.scale is set");

    // the routed experts: sigmoid gates, a selection bias, weights made to sum to one and scaled
    const MetaValue* gating = find("expert_gating_func");
    need(!gating || (int64_t) gating->u == kGatingSigmoid, "the expert gate is not a sigmoid");
    const MetaValue* shared = find("expert_shared_count");
    need(shared && shared->u == 1, "expert_shared_count is not 1");
    const MetaValue* norm = find("expert_weights_norm");
    w_norm_ = norm ? norm->u != 0 : false;
    w_scale_ = (float) num("expert_weights_scale", 0.0);
    const MetaValue& used = must("expert_used_count");
    for (int il = 0; il < n_layer_; ++il)
        if (kinds_[(size_t) il] == Kind::kMoe)
            need(of_layer(used, il) == n_used_, "the MoE layers use different numbers of experts");
    const int latent = (int) num("moe_latent_size", 0.0);
    e_embd_ = latent > 0 ? latent : n_embd_;
    max_exp_cols_ = 0;
    int fewest = n_expert_;
    for (int il = 0; il < n_layer_; ++il) {
        if (kinds_[(size_t) il] != Kind::kMoe) continue;
        const TensorInfo* g = tensor(il, "ffn_gate_inp.weight");
        need(g && g->shape.size() == 2 && (int64_t) g->shape[0] == n_embd_, "a MoE layer without its router");
        const int ne = (int) g->shape[1];
        need(ne >= n_used_ && ne <= n_expert_, "a MoE layer's experts");
        if (ne < n_expert_) max_exp_cols_ = std::max(max_exp_cols_, ne);
        fewest = std::min(fewest, ne);
    }

    // relu^2 has no limits, and a prompt batch's products guard their half sums
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
    half_sum_guard_ = true;
    std::fprintf(stderr,
                 "strata-poly: nemotron_h_moe: %d Mamba-2, %d attention and %d MoE layers; %d to %d experts a layer, "
                 "%d used, %s %d\n",
                 counts[0], counts[1], counts[2], fewest, n_expert_, n_used_, latent > 0 ? "latent" : "width",
                 e_embd_);
}

void NemotronH::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    auto none = [&](const char* n) {
        if (tensor(il, n)) throw std::runtime_error(blk(il, n) + ": not supported yet");
    };
    B.kind = kinds_[(size_t) il];
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.fixed = false;   // every launch made anew: the recordings come once the layers match llama.cpp
    switch (B.kind) {
    case Kind::kMamba:
        B.ssm_in = mat("ssm_in.weight", n_embd_, d_in_proj_);
        B.conv = f32("ssm_conv1d.weight", d_conv_, conv_ch_);
        if (tensor(il, "ssm_conv1d.bias")) B.conv_b = f32("ssm_conv1d.bias", conv_ch_);
        B.dt_b = f32("ssm_dt.bias", n_ssm_head_);
        B.a = f32("ssm_a", 1, n_ssm_head_);   // stored as -exp(A_log) by llama.cpp's converter
        B.d = f32("ssm_d", 1, n_ssm_head_);
        B.norm = f32("ssm_norm.weight", d_inner_ / n_group_, n_group_);
        B.ssm_out = mat("ssm_out.weight", d_inner_, n_embd_);
        ++n_mamba_;
        return;
    case Kind::kAttn:
        for (const char* b : {"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias"}) none(b);
        B.n_kv = max_kv_;
        B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * head_dim_);
        B.wk = mat("attn_k.weight", n_embd_, (int64_t) max_kv_ * head_dim_);
        B.wv = mat("attn_v.weight", n_embd_, (int64_t) max_kv_ * head_dim_);
        B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
        return;
    case Kind::kMoe:
        break;
    }
    // the layer's own experts: as many as its router has, as wide as the file says for it
    const TensorInfo* g = tensor(il, "ffn_gate_inp.weight");
    L.n_exp = (int) g->shape[1];
    const MetaValue* eff = meta_in(*gg_, "expert_feed_forward_length");
    L.n_ff = eff ? (int) of_layer(*eff, il) : n_ff_exp_;
    L.relu2 = true;
    load_moe(il, false);
    L.sh_up = mat("ffn_up_shexp.weight", n_embd_, n_ff_shexp_);
    L.sh_down = mat("ffn_down_shexp.weight", n_ff_shexp_, n_embd_);
    if (e_embd_ != n_embd_) {
        B.lat_down = mat("ffn_latent_down.weight", n_embd_, e_embd_);
        B.lat_up = mat("ffn_latent_up.weight", e_embd_, n_embd_);
    }
    // the selection bias, at the engine's stride: -inf past the layer's own experts, which are then never picked
    B.probs_b.assign((size_t) n_expert_, -INFINITY);
    std::fill(B.probs_b.begin(), B.probs_b.begin() + L.n_exp, 0.f);
    if (const TensorInfo* b = tensor(il, "exp_probs_b.bias")) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != L.n_exp)
            throw std::runtime_error(blk(il, "exp_probs_b.bias") + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(L.n_exp) + "]");
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) L.n_exp * sizeof(float));
    }
    if (dense_) {   // the second pass: the card's copy (the first one only counts the arena)
        B.probs_b_dev = (float*) dmalloc((size_t) n_expert_ * sizeof(float), "a router's bias");
        ck(cudaMemcpy(B.probs_b_dev, B.probs_b.data(), (size_t) n_expert_ * sizeof(float), cudaMemcpyHostToDevice),
           "a router's bias");
    }
}

size_t NemotronH::recurrent_floats() const {
    return size_t(n_mamba_) * ((size_t) (d_conv_ - 1) * conv_ch_ + (size_t) n_ssm_head_ * ssm_head_dim_ * d_state_);
}

// The attention layers' caches, and each Mamba-2 layer's convolution and scan state in state_.
void NemotronH::alloc_caches() {
    const size_t conv_n = (size_t) (d_conv_ - 1) * conv_ch_, ssm_n = (size_t) n_ssm_head_ * ssm_head_dim_ * d_state_;
    const size_t row = (size_t) max_kv_ * head_dim_ * sizeof(uint16_t);
    float* st = state_;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il) || il >= n_layer_) continue;
        if (B.kind == Kind::kMamba) {
            B.conv_st = st;
            st += conv_n;
            B.ssm_st = st;
            st += ssm_n;
        } else if (B.kind == Kind::kAttn) {
            B.keys = (uint16_t*) kv_alloc(row, 1);
            B.vals = (uint16_t*) kv_alloc(row, 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

// Per-token buffers hold rows_ tokens, row after row.
void NemotronH::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * max_kv_ * head_dim_;
    zxbcdt_ = fbuf(n * d_in_proj_);
    xbc_ = fbuf(n * conv_ch_);
    y_ = fbuf(n * d_inner_);
    q_ = fbuf(heads);
    k_ = fbuf(std::max<size_t>(kv, 1));
    v_ = fbuf(std::max<size_t>(kv, 1));
    att_ = fbuf(heads);
    score_n_ = std::max((size_t) chunk_ * n_head_ * opt_.max_context,
                        attend_group_floats(chunk_, n_head_, opt_.max_context, head_dim_));
    score_ = fbuf(score_n_);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
    // A prompt batch through llama.cpp's flash-attention kernel (fattn.hpp, heads of 128 since phase 2) in score_,
    // where it fits (Ampere or later): opt-in with STRATA_FATTN=1 until it is measured on the Nemotrons; attend_batch()
    // otherwise.  Its sums of the values run in halves (fattn_test.cu), so answers can part at near ties.
    const char* fa = std::getenv("STRATA_FATTN");
    if (rows_ >= kStageRows && fa && std::atoi(fa) == 1 && !fattn_) {   // assign_scratch() runs twice: say it once
        fattn_ = fattn_fits(head_dim_, n_head_, max_kv_, rows_, score_n_);
        std::fprintf(stderr, "strata-poly: STRATA_FATTN: prompt attention %s\n",
                     fattn_ ? "through llama.cpp's flash-attention kernel"
                            : "unchanged (no kernel for it on this card, or --prompt-batch too large for the scratch)");
    }
    xr_ = fbuf(n * n_embd_);
    xrq_ = fbuf((n * kernels::native_q8_1_bytes(n_embd_) + sizeof(float) - 1) / sizeof(float));
    sho_ = fbuf(n * n_embd_);
    if (max_exp_cols_ > 0) rlw_ = fbuf(n * max_exp_cols_);
}

// A sequence's rows are the attention layers'; the Mamba-2 layers' state is the recurrent state (save_layer()).
size_t NemotronH::card_row_bytes() const {
    size_t layers = 0;
    for (const Block& B : blocks_)
        if (B.keys) ++layers;
    return layers * 2 * (size_t) max_kv_ * head_dim_ * sizeof(uint16_t);
}

void NemotronH::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) max_kv_ * head_dim_ * sizeof(uint16_t);
    const size_t at = (size_t) first * row, bytes = (size_t) n * row;
    for (const Block& B : blocks_) {
        if (!B.keys || n == 0) continue;
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + at, bytes, cudaMemcpyDeviceToHost), "save rows");
            dst += bytes;
        }
    }
}

void NemotronH::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) max_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row;
    const size_t bytes = (size_t) n * row, skip = (size_t) (first - from) * row, all = (size_t) count * row;
    for (const Block& B : blocks_) {
        if (!B.keys || n == 0) continue;
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + at, src + skip, bytes, cudaMemcpyHostToDevice), "load rows");
            src += all;
        }
    }
}

// A Mamba-2 layer's state, convolution then scan, at its place in state_ (the recurrent part of save_state()).
void NemotronH::save_layer(int il, int64_t, uint8_t* dst) {
    const Block& B = blocks_[(size_t) il];
    if (!B.conv_st) return;
    const size_t n = (size_t) (d_conv_ - 1) * conv_ch_ + (size_t) n_ssm_head_ * ssm_head_dim_ * d_state_;
    ck(cudaMemcpy(dst + (size_t) (B.conv_st - state_) * sizeof(float), B.conv_st, n * sizeof(float),
                  cudaMemcpyDeviceToHost),
       "save state");
}

int NemotronH::head_key(int il, int n) const {
    return blocks_[(size_t) il].kind == Kind::kAttn ? room_key(n) : 0;
}

// The buffers a MoE layer's head leaves for its shared expert and its tail move with the engine's (a window's halves).
void NemotronH::shift_rows(ptrdiff_t by) {
    Engine::shift_rows(by);
    xr_ += by * n_embd_;
    xrq_ = (uint8_t*) xrq_ + by * (ptrdiff_t) kernels::native_q8_1_bytes(n_embd_);
    sho_ += by * n_embd_;
}

// The mixer: z | xBC | dt from the projection in; xBC's convolution with the state's last three inputs, its bias
// and SiLU; the scan and the D term; the gate and the grouped norm; the projection out, into ao_.  A window leaves
// the states after the tokens of snap_bits_ at snap_at_ (Engine::snap_tokens()).
void NemotronH::mamba(Block& B, int n) {
    lin(B.ssm_in, xn_, zxbcdt_, n, xq_);
    const size_t row = (size_t) d_in_proj_;
    float* snap = snap_bits_ ? snap_at_ : nullptr;
    mamba2_conv(zxbcdt_ + d_inner_, row, B.conv_st, fp(B.conv), B.conv_b.p ? fp(B.conv_b) : nullptr, xbc_, conv_ch_,
                n, st_, snap ? snap + (B.conv_st - state_) : nullptr, rec_n_, snap_bits_);
    mamba2_scan(xbc_, zxbcdt_ + d_inner_ + conv_ch_, row, fp(B.dt_b), fp(B.a), fp(B.d), B.ssm_st, y_, n_ssm_head_,
                ssm_head_dim_, d_state_, n_group_, n, st_, snap ? snap + (B.ssm_st - state_) : nullptr, rec_n_,
                snap_bits_);
    mamba2_out(y_, zxbcdt_, row, fp(B.norm), d_inner_, n_group_, n, rms_eps_, st_);
    lin(B.ssm_out, y_, ao_, n);
}

// The attention: no rotation; the keys and values into the caches; each head's softmax over the rows; the output.
void NemotronH::attn(Block& B, int64_t pos, int n) {
    lin(B.wq, xn_, q_, n, xq_);
    lin(B.wk, xn_, k_, n, xq_);
    lin(B.wv, xn_, v_, n, xq_);
    const int width = B.n_kv * head_dim_;
    store_f16(k_, B.keys, width, pos, n, st_);
    store_f16(v_, B.vals, width, pos, n, st_);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    if (n >= kStageRows) {
        if (!(fattn_ && fattn_f16(q_, B.keys, B.vals, pos, n, att_, head_dim_, n_head_, B.n_kv, scale, score_,
                                  score_n_, st_)))
            attend_batch(blas_, q_, B.keys, B.vals, pos, n, score_, score_n_, awork_, att_, head_dim_, n_head_, B.n_kv,
                         scale, st_);
    } else
        attend_group(q_, B.keys, B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_, B.n_kv, scale, st_, 0, 0,
                     false, nullptr, 0, score_n_);
    lin(B.wo, att_, ao_, n);
}

// A block from its input: a Mamba-2 or attention layer to its end; a MoE layer to its router's copy, with its
// experts' input (the latent) in xn_ / xq_ and its router's and shared expert's (the normed input) in xr_ / xrq_.
void NemotronH::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int threads = sum_threads(n_embd_);
    if (B.kind != Kind::kMoe) {
        kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
        dbg("attn_norm", il, xn_, n_embd_);
        if (B.kind == Kind::kMamba) mamba(B, n);
        else attn(B, pos_, n);
        dbg(B.kind == Kind::kMamba ? "mamba_out" : "attn_out", il, ao_, n_embd_);
        axpy(ao_, 1.f, r_, n * n_embd_, true, st_);
        return;
    }
    kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xr_, xrq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("attn_norm", il, xr_, n_embd_);
    if (e_embd_ != n_embd_) {
        lin(B.lat_down, xr_, xn_, n, xrq_);
        // the experts' rows quantized, 32 at a time (native_quantize_q8_1()'s most): a prompt batch has up to rows_
        const size_t row = kernels::native_q8_1_bytes(e_embd_);
        for (int at = 0; at < n; at += 32)
            kernels::native_quantize_q8_1(xn_ + (size_t) at * e_embd_, (uint8_t*) xq_ + (size_t) at * row, e_embd_,
                                          std::min(32, n - at), st_);
    } else {
        ck(cudaMemcpyAsync(xn_, xr_, (size_t) n * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)), "MoE");
        ck(cudaMemcpyAsync(xq_, xrq_, (size_t) n * kernels::native_q8_1_bytes(n_embd_), cudaMemcpyDeviceToDevice,
                           cs(st_)),
           "MoE");
    }
    dbg("ffn_latent_in", il, xn_, e_embd_);
    if (!batched(L, n)) moe_router(L, n);
}

// The routed experts' sum (fo_, in the latent width) up to the full width, the shared expert's output beside it,
// and into the residual.
void NemotronH::layer_tail(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    (void) L;
    if (e_embd_ != n_embd_) lin(B.lat_up, fo_, ao_, n);
    else ck(cudaMemcpyAsync(ao_, fo_, (size_t) n * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)), "MoE");
    axpy(sho_, 1.f, ao_, n * n_embd_, true, st_);
    dbg("ffn_out", il, ao_, n_embd_);
    axpy(ao_, 1.f, r_, n * n_embd_, true, st_);
}

void NemotronH::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// The router on x into rows of n_expert_: a layer with fewer experts has its scores spread to that stride, and 0 past
// them, where its selection bias is -inf.
void NemotronH::scores(const Layer& L, const float* x, int n, float* out) {
    if (L.n_exp == n_expert_) return lin(L.gate_inp, x, out, n);
    lin(L.gate_inp, x, rlw_, n);
    spread_rows(rlw_, L.n_exp, out, n_expert_, n, 0.f, st_);
}

void NemotronH::router_scores(const Layer& L, int n) { scores(L, xr_, n, rl_); }

void NemotronH::ahead_scores(const Layer& next, int n, float* out) { scores(next, xr_, n, out); }

// What ggml's MoE graph does for this model (build_moe_ffn): sigmoid gates, the n_used largest of gate + bias, and
// weights from the bare gates, made to sum to one (expert_weights_norm) and scaled (expert_weights_scale).
void NemotronH::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

bool NemotronH::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, B.probs_b_dev, nullptr, w_norm_, w_scale_, ids, w, st_);
    return true;
}

// The shared expert, relu^2 on the full width, into sho_ for layer_tail(); the routed experts' sum (fo_, latent)
// starts from nothing.
void NemotronH::moe_shared(const Layer& L, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * e_embd_ * sizeof(float), cs(st_)), "the experts' sum");
    lin(L.sh_up, xr_, g_, n, xrq_);
    relu2_rows(g_, h_, n * n_ff_shexp_, 1, st_);
    lin(L.sh_down, h_, sho_, n);
}

}  // namespace strata::poly
