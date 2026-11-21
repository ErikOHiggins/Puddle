// include/strata/poly/kernels.hpp - the float kernels of the forward passes.
//
// Every call enqueues on the given CUDA stream and returns; nothing here but blas_create allocates or waits.
// Pointers are device pointers.  The math follows llama.cpp's graphs (src/models/glm5-next.cpp, qwen35moe.cpp and
// gemma4.cpp) and the ggml ops they build, which are the reference the engine is checked against.
//
// A call works on `rows` tokens at consecutive positions.  A buffer of per-token values holds them row after row,
// each row as long as the one-token case describes.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::poly {

/// y[o] = sum_i w[o * n_in + i] * x[(o / rows_per_x) * n_in + i], o in [0, n_out).  With rows_per_x == n_out it is
/// a plain matrix-vector product; a smaller value gives each group of rows its own input (the per-head MLA
/// projections), so a token's x is n_in * n_out / rows_per_x long.
void gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream);
void gemv_bf16(const uint16_t* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows,
               void* stream);
/// The same for the many rows of a prompt batch, as a cuBLAS product on the stream of `blas` (blas_create), one
/// for each group of rows_per_x outputs.  It sums in another order than gemv_f32, so a row does not come out
/// to the last bit as it does there.  Throws std::runtime_error.
void gemm_f32(void* blas, const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows);
/// n BF16 values as floats, for gemm_f32.
void bf16_to_f32(const uint16_t* w, float* out, size_t n, void* stream);
/// Before a Q8_1 quantization whose blocks keep their sums in halves (llama.cpp's MMQ for Q4_K, Q5_K, Q2_K, Q4_1,
/// Q5_1 and IQ1_S rows; Mmq::half_sums()): each of the rows of x (ld floats apart, width values) whose sums or
/// values would leave a half's range is divided by a power of two, and factor[r] holds it (1 for the others).  A
/// SwiGLU's output can be that large: Flint's reaches 45,000, and a block of 32 summed past 65,504 is infinite.
/// The products of the divided rows are the true ones divided by the same power, to the bit: scale_rows() puts
/// them back.
void guard_half_sums(float* x, int64_t ld, int width, int rows, float* factor, void* stream);
/// The rows of y (ld floats apart, row r at row place[r] when place is given) multiplied by factor[r], or divided
/// by it; a row whose factor is 1 is not touched.
void scale_rows(float* y, int64_t ld, int width, int rows, const int32_t* place, const float* factor, bool divide,
                void* stream);

/// RMS norm over each of `rows` groups of n values: y = x / sqrt(mean(x^2) + eps) * w.  `w` (n values, shared by the
/// rows) may be null.  A row of y starts y_stride values after the one before it (0: n), so two calls can fill
/// the two halves of rows twice as long.
void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream, int y_stride = 0);
/// The threads of a block that sums over n values (rms_norm()'s among them).
int sum_threads(int n);

/// Stores the tokens' rows (n values each) as fp16 from row `pos` of the layer's cache.
void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, int rows, void* stream);
/// The same from the row the card holds at `at` when the launch runs; with `ring`, a row is its number % ring.
void store_f16_at(const float* x, uint16_t* cache, int n, const long long* at, int rows, int ring, void* stream);
/// --kv int8 (and int8k) keeps an attention cache's K and V as 8-bit codes with a delta a block of
/// kQ8_0Values values: that is llama.cpp's Q8_0 rule (delta = the block's largest magnitude / 127, code =
/// roundf(x / delta)) but NOT its block layout.  ggml's block_q8_0 is 34 bytes - a delta then 32 codes - so
/// half its blocks put the codes at an address that is 2 mod 4 and a lane's four codes can only be read
/// conservatively (memcpy into an int).  With the codes of a row back to back they are 4-aligned throughout
/// and a step is one load, and a row of a layer is n_kv heads of head_dim codes, head_dim being a multiple of
/// kQ8_0Values.  A reader carries two pointers instead of one.
inline constexpr int kQ8_0Values = 32;

/// One side of an attention cache in --kv int8's form (store_int8, attend_int8).
struct KvInt8 {
    const int8_t* code = nullptr;     ///< n_kv * head_dim codes a row, kQ8_0Values of them a block, back to back
    const uint16_t* delta = nullptr;  ///< n_kv * head_dim / kQ8_0Values halves a row, one a block
};

/// Turns an int8 cache back into halves, one kQ8_0Values-block a thread (out is the same shapes as the caches:
/// rows * n_kv * head_dim halves).  The fp16 kernels can then run over it - which is how a quantized cache is
/// used at its best, and what llama.cpp does on CUDA: its tensor-core flash attention converts a quantized
/// cache's K and V to f16 first ("other kernels convert these to f16"), and its Metal backend gained a dedicated
/// pass for the same thing.  It costs one O(rows) pass to get the attention back onto the fast path.
void dequant_int8_to_f16(const int8_t* code, const uint16_t* delta, uint16_t* out, int64_t n_blocks, void* stream);

/// A runtime capability probe for the int8 tensor-core path: it runs one m16n8k32 on this card and checks the
/// result against the same product computed on the host, so a kernel can be dispatched on the answer rather than
/// on an assumed architecture.  True on sm_80 and up; a backend whose int8 mma is not there (HIP without
/// mfma_i32_16x16x32_i8, a card before Ampere) answers false and the caller keeps dp4a.
bool probe_mma_layout();

/// Whether attend_int8 can take its tensor-core tier here: the m16n8k32 int8 step needs Ampere (sm_80), where the
/// fp16 tensor-core kernels need only Turing.  False under HIP, before Ampere, and on an older card's PTX.
bool int8_tensor_cores();

/// A study, not part of the engine: the same 16x8x256 product through the tensor cores and through dp4a, in ms.
void probe_mma_speed(double* mma_ms, double* dp4a_ms);

/// Stores the tokens' latents (n_kv * head_dim values a row, every kQ8_0Values of them a block) as codes and their
/// deltas from row `pos` of the layer's cache; `at` and `ring` as in store_f16_at.  Half the bytes a token takes.
void store_int8(const float* x, int8_t* code, uint16_t* delta, int n_kv, int head_dim, int64_t pos, int rows,
                void* stream);
void store_int8_at(const float* x, int8_t* code, uint16_t* delta, int n_kv, int head_dim, int rows,
                   const long long* at, int ring, void* stream);

/// Stores the tokens' rows as ggml Q8_0 (one fp16 scale a 32-value block of int8 codes) from row `pos`.
/// codes: n int8 a row; scales: n/32 fp16 a row.  n must be a multiple of 32 (QK8_0).
void store_q8(const float* x, int8_t* codes, uint16_t* scales, int n, int64_t pos, int rows, void* stream);
/// The same from the row the card holds at `at` when the launch runs; with `ring`, a row is its number % ring.
void store_q8_at(const float* x, int8_t* codes, uint16_t* scales, int n, const long long* at, int rows, int ring,
                 void* stream);
/// A cuBLAS handle that enqueues on `stream`, for gemm_f32.  Throws std::runtime_error.
void* blas_create(void* stream);
void blas_destroy(void* blas);

/// out = silu(min(gate, limit)) * clamp(up, -limit, limit).
void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream);
/// The global scales of NVFP4 routed experts converted from NVIDIA's checkpoints (the GGUF's ffn_gate_exps.scale
/// and ffn_up_exps.scale, one an expert), on the card.  Row r of an activation is expert ids[r]'s, or e[r]'s when
/// ids is null: its gate values are multiplied by gate[that expert] and its up values by up[it] first.
struct ExpertScales {
    const float* gate = nullptr;   ///< null: none (1)
    const float* up = nullptr;     ///< null: none (1)
    const int32_t* ids = nullptr;
    int32_t e[32] = {};
};
/// swiglu_clamp for rows that each hold n gate values and then n up values; out has n values a row.
void swiglu_clamp_rows(const float* gate_up, float* out, int n, int rows, float limit, void* stream,
                       const ExpertScales* scales = nullptr);
/// relu(up)^2 for rows that each hold n up values where a gate|up row has its gate (rows 2n apart: the routed experts'
/// buffers are shared with swiglu_clamp_rows), n values a row into out: Nemotron-H's experts, up and down alone
/// (llama.cpp's LLM_FFN_RELU_SQR).  The scales' gate is up's (Layer::relu2).
void relu2_rows(const float* up, float* out, int n, int rows, void* stream, const ExpertScales* scales = nullptr);
/// out = gelu(gate) * up, with the tanh form of GELU that ggml's CUDA backend computes.
void geglu(const float* gate, const float* up, float* out, int n, void* stream);
/// geglu for rows that each hold n gate values and then n up values; out has n values a row.
void geglu_rows(const float* gate_up, float* out, int n, int rows, void* stream, const ExpertScales* scales = nullptr);
/// w[r] *= scale[ids[r]] for `rows` weights: the down scales of ExpertScales' experts, on their routing weights.
void expert_weights(float* w, const float* scale, const int32_t* ids, int rows, void* stream);
/// The routing of `rows` tokens from their router scores (n_expert a token, times in_scale), as an architecture's
/// route() does it on the host: probabilities by `gate` (0 softmax, 1 sigmoid, 2 sqrt(softplus), 3 sigmoid with the
/// pick on the bare scores), the n_used largest of the probabilities plus `bias` (null: none), the lower expert
/// first among equals; their weights are the probabilities, with `norm` divided by their sum (ggml's floor under
/// it), times escale[expert] (null: none), times w_scale (0: not).  ids and w: n_used a token, on the card.  Gate 3
/// is llama.cpp's SIGMOID_LOGIT_ADD, Kolibri 1's: the weights are sigmoid(scores) and the pick is scores + bias.
/// Throws std::invalid_argument past 512 experts.
void route_rows(const float* logits, int n_expert, int n_used, int rows, int gate, float in_scale, const float* bias,
                const float* escale, bool norm, float w_scale, int32_t* ids, float* w, void* stream);
/// y's rows of `stride` values from x's rows of n (n <= stride), the rest of each row `fill`: a router's scores of a
/// layer with fewer experts than the model's largest, at the engine's stride (Nemotron 3.5 Bolt).
void spread_rows(const float* x, int n, float* y, int stride, int rows, float fill, void* stream);
/// acc[t] += sum over k < n_used of w[t * n_used + k] * x[t * n_used + k], for rows of n values: a token's
/// weighted experts added to its row.
void moe_sum(const float* x, const float* w, float* acc, int n, int n_used, int rows, void* stream);
/// acc = x * a, or acc += x * a when `add`.
void axpy(const float* x, float a, float* acc, int n, bool add, void* stream);
/// y = x for two runs of 4-byte words, n0 and n1 of them, in one launch.  Pinned host memory is the card's to read
/// and write too, and a kernel's copy to or from it has none of the wait that a copy engine's has on Windows,
/// some 25 microseconds between it and the launches beside it.
void copy2(const void* x0, void* y0, int n0, const void* x1, void* y1, int n1, void* stream);
/// Up to 32 vectors of n values, each times its weight into its own sum, one after the other: acc[k] += x[k] * w[k]
/// for k = 0, 1, ...  Two of them may have the same sum.
struct ScaledRows {
    const float* x[32];
    float* acc[32];
    float w[32];
};
void add_scaled(const ScaledRows& rows, int count, int n, void* stream);

// ---- attention over a key and value cache (qwen35moe, gemma4), and the gated delta net with one decay a head

/// For `rows` tokens of `heads` heads each: a head's RMS norm with the weights w (dim values, the same for every
/// head; null: none), then ggml's rotary embedding on its first n_rot values in the NeoX pairing: value i goes
/// with value i + n_rot / 2, and the pair turns by position * theta_scale^i, divided by freq[i] when freq is
/// given.  A head's input starts x_stride floats after the one before it; y has dim values a head.
/// The position is the token's place in the call counted from `pos`.  With `axes` (three numbers a token) it is
/// the token's position on one of three axes instead: pair i takes axis (i % sections) % 3, ggml's interleaved
/// rope_multi, which is how a picture's rows get a line and a column.
/// With `at`, the first token's position is not `pos` but what the card holds at `at` when the launch runs: a
/// launch recorded once can then be replayed for every position (attend() and store_f16_at() take it too).
void head_norm_rope(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                    float theta_scale, int64_t pos, int rows, float eps, void* stream, const float* freq = nullptr,
                    const int32_t* axes = nullptr, int sections = 1, const long long* at = nullptr);
/// YaRN as ggml's rope_yarn computes it, for head_norm_rope_yarn() (laguna's full layers).  Pair i's angle is the
/// interpolated one, freq_scale times the angle head_norm_rope() gives it, mixed with that angle itself by the
/// ramp ext_factor * (1 - clamp((i - corr_low) / max(0.001, corr_high - corr_low), 0, 1)).  mscale multiplies the
/// rotated values' cos and sin: the final magnitude, the attention factor llama.cpp passes times the
/// 1 + 0.1 * ln(1 / freq_scale) ggml adds when ext_factor is not 0.  The values past n_rot are only normed.
struct RopeYarn {
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float corr_low = 0.0f, corr_high = 0.0f;   ///< ggml_rope_yarn_corr_dims()
    float mscale = 1.0f;
};
/// head_norm_rope() with YaRN: the same norm, NeoX pairing, positions and `at`, with no frequency factors or
/// axes.  CUDA and HIP only.
void head_norm_rope_yarn(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                         float theta_scale, int64_t pos, int rows, float eps, void* stream, const RopeYarn& yarn,
                         const long long* at = nullptr);
/// An attention cache stored as ggml Q8_0: a row is `n` int8 codes and `n / 32` fp16 scales with
/// `n = kv_heads * dim`, row-major (`codes[row * n + blk * 32 + j]`, `scales[row * (n / 32) + blk]`).
/// A q8 read is active wherever `codes` is not null.  The value a block holds is
/// `(float)codes[..] * __half2float(scales[..])` with one scale a 32-value block.
struct Kvq8 {
    const int8_t* codes = nullptr;
    const uint16_t* scales = nullptr;
};

/// Causal attention for the tokens from position `pos`: token t attends the rows 0 to pos + t, or with `window`
/// only the last `window` of them.  q: n_head * dim a token.  kcache and vcache: fp16, kv_heads * dim a row; the
/// heads read a row's keys and values in turn, n_head / kv_heads heads each.  With `ring`, the sequence's row p
/// is cache row p % ring.  out: n_head * dim a token.  It takes `chunk` tokens a launch, which read the caches
/// once between them, and score is scratch of chunk * n_head floats for each row a launch's tokens attend
/// between them: pos + rows, or window - 1 + chunk if that is less.  A token comes out the same whatever tokens
/// the call has beside it.  Queries and keys are read four values at a time: dim is a multiple of 4, q lies at
/// a multiple of 16 bytes and kcache of 8, or it throws std::invalid_argument.
/// With `whole` the call's tokens see each other: each attends up to the call's last row, so a token can attend
/// window - 1 + rows of them.
/// With `at`, the tokens start at the position the card holds there; the launches are sized for `room` rows,
/// which must be no fewer than a launch's tokens attend between them, and score has room for them.  With `whole`
/// too the call is one launch: rows is no more than chunk.
/// score_floats: the floats score holds, when more than the above.  Heads of 64 values in groups of 4, or of 128 in
/// groups of 8, over the whole conversation then take a launch block each block of rows, a launch as many tokens as
/// score has room for those blocks' sums of at the longest the call can attend (attend_split_floats() for chunk of
/// them); the results differ from the other kernels' in the last bits.
/// With `kq8` or `vq8` (codes not null) that cache is read as Q8_0 rather than fp16: its codes/scale pointers
/// follow the same row and head index the fp16 cache would, and a block of 32 values dequantizes as
/// `(float)codes[..] * __half2float(scales[..])`.  The fp16 path is unchanged where they are left null.
/// With `sinks` (n_head floats on the card, flint's), each head's softmax has one more logit, its sink, which
/// weighs no value: it only takes weight from the rows.
void attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows, float* score,
            int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window = 0,
            int ring = 0, bool whole = false, const long long* at = nullptr, int room = 0, size_t score_floats = 0,
            Kvq8 kq8 = {}, Kvq8 vq8 = {}, const float* sinks = nullptr, size_t part_floats = 0);
/// The floats of score that attend()'s blocks of rows take for calls of up to chunk tokens attending up to max_rows
/// rows (score_floats): with that much a model's every call takes them.
size_t attend_split_floats(int chunk, int n_head, int max_rows, int dim);
/// attend() for heads of 128 values in groups of 2 to 7 (Laguna's full layers, 6), over an fp16 cache or a q8_0 one
/// (kq8 and vq8 both): a launch block each block of 256 rows, its sums added up by a second launch, the group's
/// heads in 8 places.  The same arguments as attend(), and its results to the last bits (the sums go in another
/// order); a token comes out the same whatever tokens the call has beside it.  Any other call (another shape, a
/// window, a ring, `whole`, or score without attend_group_floats() of room) goes to attend() as it is.  CUDA and
/// HIP only.
void attend_group(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows, float* score,
                  int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window = 0,
                  int ring = 0, bool whole = false, const long long* at = nullptr, int room = 0,
                  size_t score_floats = 0, Kvq8 kq8 = {}, Kvq8 vq8 = {}, const float* sinks = nullptr);
/// The floats of score that attend_group() takes for calls of up to chunk tokens attending up to max_rows rows.
size_t attend_group_floats(int chunk, int n_head, int max_rows, int dim);
/// attend() for a prompt batch: the same attention with cuBLAS products on the tensor cores, several times
/// faster on many tokens.  The products read the cache's halves where they lie, take the queries and the
/// softmax's weights rounded to halves, and sum floats.  A token's result depends on the tokens beside it, so
/// the windows of guessed tokens use attend().
/// It takes the call's tokens together and the rows they attend a block at a time: as many rows as have their
/// scores and weights, 6 bytes for each of a token's n_head heads, in the score_floats floats of `score`, which
/// must hold 2 * n_head floats.  work: rows * n_head * (dim + dim / 2 + 3) + 1 floats.  Throws
/// std::runtime_error.
/// Heads of 64 values in groups of 4 over the whole conversation take one kernel instead, which keeps the scores in
/// registers, on NVIDIA cards from Turing on; it uses neither blas, score nor work.  So do heads of 128 values in
/// groups of 8, with a window, a ring and sinks too, from Ampere on; a token's keys go in blocks of 64 counted from
/// the cache's first row, so its result is the same whatever tokens the call has beside it.
/// With `kq8` or `vq8` (codes not null) that cache is read as Q8_0 like attend(): before each block of `m` keys
/// the product reads, its rows are dequantized into the fp16 workspace `kwork` / `vwork`, and the products then
/// read that exactly as they read an fp16 cache, so the results match attend()'s q8 read.  Each workspace holds
/// work_halves halves, `work_halves / (kv_heads * dim)` rows at most: a block of keys larger than that is split,
/// taking more products.  The fp16 path (both null) is unchanged and reads the caches where they lie.
/// `sinks` as attend()'s: the sink joins each token's sum after the last block of rows.
void attend_batch(void* blas, const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows,
                  float* score, size_t score_floats, float* work, float* out, int dim, int n_head, int kv_heads,
                  float scale, void* stream, int window = 0, int ring = 0, bool whole = false, Kvq8 kq8 = {},
                  Kvq8 vq8 = {}, uint16_t* kwork = nullptr, uint16_t* vwork = nullptr, size_t work_halves = 0,
                  const float* sinks = nullptr);
/// attend() over int8 caches (store_int8): the same attention, the same order of products and sums.  `q8` is scratch
/// in the same form for this call's queries (rows * n_head * dim / kQ8_0Values blocks of codes, and as many
/// deltas).  A score is a dp4a of a lane's own run of codes - the query's against the key's, both 4-aligned - and
/// the block's two deltas scale the integer sum once: llama.cpp's vec_dot_q8_0_q8_1, whose whole point is that
/// neither side is ever turned into floats.  No transposed-value path (a decode-time call has few tokens and reads
/// the weights from the scores' buffer).  `at` (a recorded launch) and `room` as in attend().
void attend_int8(const float* q, KvInt8 q8, KvInt8 k, KvInt8 v, int64_t pos, int rows, float* score, int chunk,
                 float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window = 0, int ring = 0,
                 bool whole = false, const long long* at = nullptr, int room = 0);
/// attend_int8 with the values' side left in fp16 (--kv int8k): the scores are the int8 ones against the keys'
/// codes, the weighted sums read the fp16 cache the usual way.  `v_h` is the values' cache, `n_kv * dim`
/// halves a row, the ring's included.
void attend_int8_mixed(const float* q, KvInt8 q8, KvInt8 k, const uint16_t* v_h, int64_t pos, int rows,
                       float* score, int chunk, float* out, int dim, int n_head, int kv_heads, float scale,
                       void* stream, int window = 0, int ring = 0, bool whole = false,
                       const long long* at = nullptr, int room = 0);
/// x = cap * tanh(x / cap).
void softcap(float* x, size_t n, float cap, void* stream);
/// The first of the largest of n logits and the softmax's probability of it, summed without the logits more than
/// 20 below it.
struct Likeliest {
    int32_t token;
    float prob;
};
void likeliest(const float* logits, int n, Likeliest* out, void* stream);
/// acc[r][i] *= sigmoid(g[r][i]) for `rows` groups of n values; group r's gates start r * g_stride floats into g.
/// With one_gate a group has one gate for all its values.
void mul_sigmoid(float* acc, const float* g, int n, int rows, int g_stride, bool one_gate, void* stream);
/// mul_sigmoid() with ggml's softplus in place of the sigmoid: g above 20 as it is, else log1p(exp(g)) (laguna's
/// attention output gate).  CUDA and HIP only.
void mul_softplus(float* acc, const float* g, int n, int rows, int g_stride, bool one_gate, void* stream);
/// A causal convolution of four taps over each of `channels` channels, and its SiLU.  x and out: `channels` values
/// a token.  state: 3 past inputs a channel, oldest first, advanced here by `rows` tokens.  w: (4, channels).
/// With snap, the state as it is after some of the call's tokens is also written there, for a window of guessed
/// tokens to go back to (gdn_step too): `snaps` has a bit for each token that leaves it, and the states lie in
/// the tokens' order, `stride` floats apart.
void conv4_silu(const float* x, float* state, const float* w, float* out, int channels, int rows, void* stream,
                float* snap = nullptr, size_t stride = 0, unsigned snaps = 0);
/// The delta-rule step of every value head and its gated output, token after token: ggml's gated_delta_net op
/// with one decay a head.  qkv: a token's conv outputs, the k_heads key heads' q, their k, then the n_head value
/// heads' v, dim values each; value head h reads key head h % k_heads.  alpha and beta: n_head a token, as their
/// projections give them.  dt, a: n_head.  The state decays by exp(softplus(alpha + dt) * a) and takes
/// sigmoid(beta) of the error.  s: n_head * dim * dim, updated.  z: n_head * dim a token.
/// out = rms_norm(o) * o_norm * silu(z), n_head * dim values a token.  qkv, alpha and beta are overwritten.
void gdn_step(float* qkv, float* alpha, float* beta, const float* dt, const float* a, float* s,
              const float* z, const float* o_norm, float* out, int dim, int n_head, int k_heads, int rows,
              float norm_eps, void* stream, float* snap = nullptr, size_t stride = 0, unsigned snaps = 0);

}  // namespace strata::poly
