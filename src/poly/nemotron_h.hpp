// src/poly/nemotron_h.hpp - NVIDIA's Nemotron-H with routed experts (llama.cpp architecture "nemotron_h_moe"):
// Nemotron 3.5 Bolt 74B-A9B, and Super 120B-A12B after it.
//
// One residual stream, and in each block one of three things after its RMS norm, by what the file says of the layer
// (llama.cpp's is_recr(): no key-value heads and no FFN width is a Mamba-2 layer; no FFN width an attention layer):
// - a Mamba-2 mixer: a projection in to z | xBC | dt, a causal convolution of four taps over xBC with the state's
//   last three inputs, the selective scan with one decay a head and a state of d_state values for each value of
//   each head (groups of heads share B and C), the D term, the gate silu(z), an RMS norm in groups, a projection out
//   (include/strata/poly/mamba2.hpp);
// - attention without a rotation (NoPE), its keys and values in a cache;
// - routed experts beside a shared expert: a sigmoid router on the block's normed input picks by its gates and a
//   selection bias, and weighs the picks by their bare gates made to sum to one and scaled (Laguna's router); the
//   experts work in a latent width (LatentMoE: a projection down before them and up after them, Engine::e_embd_),
//   each up and down alone with relu^2 between (Layer::relu2); the shared expert is the same on the full width.
// Bolt's MoE layers each have their own number of experts and width (Layer::n_exp, n_ff).
//
// Super's file has a block after its layers that guesses the token after the next (its MTP head, llama.cpp's
// graph_mtp in src/models/nemotron-h-moe.cpp): the token's embedding and the model's output, each RMS normed, side by
// side through a projection (Engine::draft_input()); an attention sublayer; routed experts as a MoE layer's, after
// their own norm (post_attention_norm); a LayerNorm, and the model's output matrix.  The block's output stands in for
// the model's in the guesses after the first (draft_step()).
#pragma once

#include "engine.hpp"

#include <algorithm>

namespace strata::poly {

class NemotronH final : public Engine {
public:
    /// Its pictures: strata-vision's rows (llama.cpp's mtmd, nemotron_omni) where the server put them, between <img> and
    /// </img>.  They are read as text is (causal, no rotation), so a picture may take several calls (Engine::whole()).
    bool sees() const override { return true; }
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    void check_draft_block() override;
    void draft_step(int32_t token, int i) override;
    void block_attend(int64_t pos, int n) override;
    void block_rows(int64_t pos, int n) override;
    size_t recurrent_floats() const override;
    bool snaps_spread() const override { return true; }
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override { return std::max(d_inner_, n_head_ * head_dim_); }
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    void router_scores(const Layer& L, int n) override;
    void ahead_scores(const Layer& next, int n, float* out) override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    int head_key(int il, int n) const override;
    void shift_rows(ptrdiff_t by) override;
    const Names* meta_alias(const char* key) const override;

private:
    enum class Kind : uint8_t { kMamba, kAttn, kMoe };
    /// A layer's weights beside the ones every architecture has (Layer), and its state or caches.
    struct Block {
        Kind kind = Kind::kMamba;
        // Mamba-2
        Dev ssm_in, conv, conv_b, dt_b, a, d, norm, ssm_out;
        float* conv_st = nullptr;   ///< in state_: 3 past inputs a channel of xBC
        float* ssm_st = nullptr;    ///< in state_: n_ssm_head_ * ssm_head_dim_ * d_state_
        // attention
        Dev wq, wk, wv, wo;
        uint16_t* keys = nullptr;   ///< rows of n_kv * head_dim, fp16, max_context of them
        uint16_t* vals = nullptr;
        int n_kv = 0;
        // MoE
        Dev lat_down, lat_up;
        Dev post_norm;   ///< the draft block's: its routed experts' norm, after its attention
        std::vector<float> probs_b;   ///< the selection bias, n_expert_ of it: -inf past the layer's own experts
        float* probs_b_dev = nullptr;
    };

    void mamba(Block& B, int n);
    void attn(Block& B, int64_t pos, int n);
    /// Layer L's router on x (n rows of n_embd_) into out, n rows of n_expert_: the layer's own experts' scores, and
    /// 0 past them (their selection bias keeps them out).
    void scores(const Layer& L, const float* x, int n, float* out);

    std::vector<Block> blocks_;   ///< one a layer of layers_
    std::vector<Kind> kinds_;     ///< per layer, from the file
    int n_mamba_ = 0;             ///< this engine's Mamba-2 layers (recurrent_floats())
    // Mamba-2
    int d_conv_ = 0, d_inner_ = 0, d_state_ = 0, n_group_ = 0, n_ssm_head_ = 0, ssm_head_dim_ = 0;
    int conv_ch_ = 0, d_in_proj_ = 0;   ///< xBC's channels, and ssm_in's outputs: z | xBC | dt
    // attention
    int n_head_ = 0, head_dim_ = 0, max_kv_ = 0;
    int chunk_ = 1;
    // MoE
    bool w_norm_ = false;
    float w_scale_ = 0;
    float ln_eps_ = 0;            ///< the draft block's LayerNorm (attention.layer_norm_epsilon)
    int max_exp_cols_ = 0;        ///< the most experts of a layer whose experts are fewer than n_expert_

    // device scratch
    float *zxbcdt_ = nullptr, *xbc_ = nullptr, *y_ = nullptr;           ///< a Mamba-2 layer's
    float *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr, *score_ = nullptr, *awork_ = nullptr;
    size_t score_n_ = 0;
    bool fattn_ = false;   ///< a prompt batch through fattn_f16() (fattn.hpp); STRATA_FATTN=0: never
    bool fdec_ = false;    ///< a few tokens through fdec_f16() (fattn.hpp); STRATA_FDEC=0: never
    int fdec_from_ = 4096;   ///< from this position (STRATA_FDEC=1: 0)
    bool fdec_said_ = false;
    float* ssd_ = nullptr;   ///< mamba2_scan_ssd()'s scratch, null where it is not used (STRATA_MAMBA2_SSD=0)
    size_t ssd_n_ = 0;
    bool ssd_said_ = false;
    float* xr_ = nullptr;     ///< a MoE layer's normed input on the full width: its router's and shared expert's input
    void* xrq_ = nullptr;     ///< and its Q8_1 rows
    float* sho_ = nullptr;    ///< the shared expert's output, for layer_tail()
    float* rlw_ = nullptr;    ///< a router's scores before they are spread to n_expert_ a row (scores())
};

}  // namespace strata::poly
