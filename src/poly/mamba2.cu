// src/poly/mamba2.cu - the Mamba-2 mixer's kernels (include/strata/poly/mamba2.hpp): what llama.cpp's
// ggml_ssm_conv, ggml_ssm_scan (Mamba-2's form: one decay a head), the gate and the grouped RMS norm compute, with
// the recurrent state kept on the card and snapshots for a window of guessed tokens.  Tested by mamba2_test.cu
// against a host copy of llama.cpp's arithmetic (ggml-cpu ops.cpp, 08c252e22).
#include "strata/poly/mamba2.hpp"

#include <cuda_runtime.h>
#if !defined(STRATA_USE_HIP)
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cub/block/block_scan.cuh>
#endif

#include <algorithm>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

constexpr int WARP = 32;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

// One value a thread summed over the block; `part` has a float a warp.
__device__ __forceinline__ float block_sum(float x, float* part) {
    x = warp_sum(x);
    if ((threadIdx.x & (WARP - 1)) == 0) part[threadIdx.x / WARP] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) s += part[k];
    __syncthreads();
    return s;
}

// ggml's silu (x / (1 + e^-x)) and softplus (ggml-impl.h: above 20 the input itself)
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }

// Where a token leaves its state (mamba2.hpp): null for a token that leaves none.
__device__ __forceinline__ float* snap_of(float* snap, size_t stride, unsigned snaps, int row) {
    return row < 32 && (snaps >> row & 1u) ? snap + size_t(__popc(snaps & ((1u << row) - 1u))) * stride : nullptr;
}

// The four taps in ggml_ssm_conv's order (a sum from 0, oldest input first), then the bias, then SiLU.
__device__ __forceinline__ float conv_out(float s0, float s1, float s2, float v, const float* wc, float b) {
    float y = 0.0f;
    y += s0 * wc[0];
    y += s1 * wc[1];
    y += s2 * wc[2];
    y += v * wc[3];
    return silu(y + b);
}

// A thread a channel, walking the call's tokens; the state after a token of `snaps` goes to its snapshot.
__global__ void conv_walk_kernel(const float* __restrict__ in, size_t in_stride, float* __restrict__ state,
                                 const float* __restrict__ w, const float* __restrict__ bias, float* __restrict__ out,
                                 int channels, int rows, float* __restrict__ snap, size_t stride, unsigned snaps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* wc = w + 4 * size_t(c);
    const float b = bias ? bias[c] : 0.0f;
    float* st = state + size_t(c) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    for (int row = 0; row < rows; ++row) {
        const float v = in[size_t(row) * in_stride + c];
        out[size_t(row) * channels + c] = conv_out(s0, s1, s2, v, wc, b);
        s0 = s1;
        s1 = s2;
        s2 = v;
        if (float* m = snap_of(snap, stride, snaps, row)) {
            m += size_t(c) * 3;
            m[0] = s0;
            m[1] = s1;
            m[2] = s2;
        }
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
}

// From this many tokens (and no snapshots) a thread a channel of a token: the same sum, all tokens at once.
constexpr int CONV_ROWS = 16;

__global__ void conv_rows_kernel(const float* __restrict__ in, size_t in_stride, const float* __restrict__ state,
                                 const float* __restrict__ w, const float* __restrict__ bias, float* __restrict__ out,
                                 int channels, int row0) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int row = row0 + int(blockIdx.y);
    const float* st = state + size_t(c) * 3;
    auto at = [&](int r) { return r >= 0 ? in[size_t(r) * in_stride + c] : st[3 + r]; };
    out[size_t(row) * channels + c] =
        conv_out(at(row - 3), at(row - 2), at(row - 1), at(row), w + 4 * size_t(c), bias ? bias[c] : 0.0f);
}

// The state after the call: its last three inputs, the state's own where the call has fewer.
__global__ void conv_state_kernel(const float* __restrict__ in, size_t in_stride, float* __restrict__ state,
                                  int channels, int rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float* st = state + size_t(c) * 3;
    auto at = [&](int r) { return r >= 0 ? in[size_t(r) * in_stride + c] : st[3 + r]; };
    const float a = at(rows - 3), b = at(rows - 2), d = at(rows - 1);
    st[0] = a;
    st[1] = b;
    st[2] = d;
}

// The scan: a warp a value of a head (head h, value i), its d_state = PER * 32 state values in registers, PER a lane,
// for the whole walk over the tokens.  A token: dt' = softplus(dt + dt_bias), dA = exp(dt' * A), the state takes
// state * dA + B * (x * dt'), and y = C . state + D * x (ggml_ssm_scan, then build_mamba2_layer's D term).
template <int PER>
__global__ void __launch_bounds__(256)
scan_kernel(const float* __restrict__ xbc, const float* __restrict__ dt, size_t dt_stride,
            const float* __restrict__ dt_bias, const float* __restrict__ a, const float* __restrict__ d,
            float* __restrict__ state, float* __restrict__ y, int n_head, int head_dim, int n_group, int rows,
            float* __restrict__ snap, size_t stride, unsigned snaps) {
    constexpr int DS = PER * WARP;
    const int wid = int((size_t(blockIdx.x) * blockDim.x + threadIdx.x) / WARP);
    const int lane = int(threadIdx.x & (WARP - 1));
    if (wid >= n_head * head_dim) return;   // a whole warp
    const int h = wid / head_dim, g = h / (n_head / n_group);
    const size_t d_inner = size_t(n_head) * head_dim, row_n = d_inner + 2 * size_t(n_group) * DS;
    const size_t mine = size_t(wid) * DS;
    float* st = state + mine;
    float s[PER];
#pragma unroll
    for (int k = 0; k < PER; ++k) s[k] = st[lane + k * WARP];
    const float ah = a[h], dh = d[h], bh = dt_bias[h];
    for (int row = 0; row < rows; ++row) {
        const float* x = xbc + size_t(row) * row_n;
        const float* B = x + d_inner + size_t(g) * DS;
        const float* C = x + d_inner + size_t(n_group + g) * DS;
        const float xv = x[wid];
        const float dts = softplus(dt[size_t(row) * dt_stride + h] + bh);
        const float dA = expf(dts * ah);
        const float xdt = xv * dts;
        float* m = snap_of(snap, stride, snaps, row);
        if (m) m += mine;
        float o = 0.0f;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int j = lane + k * WARP;
            const float n = s[k] * dA + B[j] * xdt;
            s[k] = n;
            if (m) m[j] = n;
            o += n * C[j];
        }
        o = warp_sum(o);
        if (lane == 0) y[size_t(row) * d_inner + size_t(wid)] = o + xv * dh;
    }
#pragma unroll
    for (int k = 0; k < PER; ++k) st[lane + k * WARP] = s[k];
}

// A block a token's group: silu(z) * y, then its RMS norm and weights.
__global__ void out_kernel(float* __restrict__ y, const float* __restrict__ z, size_t z_stride,
                           const float* __restrict__ norm, int d_inner, int per_group, int row0, float eps) {
    __shared__ float part[32];
    const size_t row = size_t(row0) + blockIdx.y, first = size_t(blockIdx.x) * per_group;
    float* yr = y + row * d_inner + first;
    const float* zr = z + row * z_stride + first;
    const float* nr = norm + first;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < per_group; c += blockDim.x) {
        const float v = silu(zr[c]) * yr[c];
        yr[c] = v;
        sum += v * v;
    }
    sum = block_sum(sum, part);
    const float scale = 1.0f / sqrtf(sum / float(per_group) + eps);
    for (int c = threadIdx.x; c < per_group; c += blockDim.x) yr[c] = yr[c] * scale * nr[c];
}

inline cudaStream_t cs(void* stream) { return static_cast<cudaStream_t>(stream); }
inline int blocks(size_t n, int block) { return int((n + block - 1) / block); }
constexpr int MAX_GRID_Y = 65535;

#if !defined(STRATA_USE_HIP)
// ---- the scan of a prompt batch in llama.cpp's SSD form (ggml-cuda ssm-scan.cu at 08c252e22, ssm_scan_ssd_f32_cuda):
// the tokens in chunks of 256; within a chunk the scan is the products
//   y  = (S^T C') + (X_dt M^T)       M = the causal decay between the chunk's tokens times C B^T
//   S' = decay(chunk) S + B' X_dt^T  B' = B decayed to the chunk's end, C' = C decayed from its start
// on the tensor cores, and only the state goes from chunk to chunk.  X_dt, B' and M are halves, as in llama.cpp;
// its fp32 products (C B^T and S^T C') run in TF32, as its cuBLAS handle does (CUBLAS_TF32_TENSOR_OP_MATH).
constexpr int SSD_CHUNK = 256;                                          // SSM_SSD_CHUNK_SIZE
constexpr int SSD_DT_BLOCK = 256, SSD_DT_ITEMS = 32;                    // SSM_SSD_DT_BLOCK / _MAX_ITEMS
constexpr int SSD_MIN_ROWS = 128, SSD_MAX_ROWS = SSD_DT_BLOCK * SSD_DT_ITEMS;   // SSM_SSD_MIN / MAX_TOKENS

// softplus(dt + dt_bias) of each token of head blockIdx.x, and its running sum over the call's tokens
// (ssm_ssd_prepare_dt_kernel; the bias is llama.cpp's graph's ggml_add before the scan)
template <int BLOCK, int ITEMS>
__global__ void ssd_dt_kernel(const float* __restrict__ dt, size_t dt_stride, const float* __restrict__ dt_bias,
                              float* __restrict__ dt_sp, float* __restrict__ csum, int n_head, int n_tok) {
    const int h = blockIdx.x;
    const float bh = dt_bias[h];
    const int items = (n_tok + BLOCK - 1) / BLOCK;
    float v[ITEMS];
    for (int i = 0; i < items; ++i) {
        const int t = i * BLOCK + threadIdx.x;
        float sp = 0.0f;
        if (t < n_tok) {
            const float x = dt[size_t(t) * dt_stride + h] + bh;
            sp = x <= 20.0f ? log1pf(expf(x)) : x;
            dt_sp[size_t(t) * n_head + h] = sp;
        }
        v[i] = sp;
    }
    using Scan = cub::BlockScan<float, BLOCK>;
    __shared__ typename Scan::TempStorage tmp;
    __shared__ float step;
    float run = 0.0f;
    for (int i = 0; i < items; ++i) {
        float inc;
        Scan(tmp).InclusiveSum(v[i], inc);
        const int t = i * BLOCK + threadIdx.x;
        if (t < n_tok) csum[size_t(t) * n_head + h] = run + inc;
        if (threadIdx.x == BLOCK - 1) step = inc;
        __syncthreads();
        run += step;
        __syncthreads();   // tmp and step are used again
    }
}

// A chunk's inputs to the products for head blockIdx.y (ssm_ssd_pre_matmul_kernel): X_dt = x dt (head_dim
// fastest), B' = B decayed to the chunk's end, C' = C decayed from its start (d_state fastest), and M = the decay
// from token t_in to t_out times (C B^T)[t_out, t_in] for t_in <= t_out, 0 above (t_out fastest).
template <int BLOCK>
__global__ void ssd_pre_kernel(const float* __restrict__ csum, const float* __restrict__ dt_sp,
                               const float* __restrict__ a, const float* __restrict__ xbc, size_t row_n, int d_inner,
                               half* __restrict__ x_dt, half* __restrict__ b_w, float* __restrict__ c_s,
                               const float* __restrict__ cb, half* __restrict__ m, int len, int head_dim, int n_head,
                               int n_group, int d_state, int first) {
    const int h = blockIdx.y, g = h / (n_head / n_group);
    const float ah = a[h];
    const int idx = blockIdx.x * BLOCK + threadIdx.x;
    const float base = first > 0 ? csum[size_t(first - 1) * n_head + h] : 0.0f;
    const float last = csum[size_t(first + len - 1) * n_head + h] - base;
    const int n_xdt = len * head_dim;
    if (idx < n_xdt) {
        const int i = idx % head_dim, t = idx / head_dim;
        const float xv = xbc[size_t(first + t) * row_n + size_t(h) * head_dim + i];
        x_dt[size_t(h) * n_xdt + idx] = __float2half(xv * dt_sp[size_t(first + t) * n_head + h]);
    }
    const int n_bw = d_state * len;
    if (idx < n_bw) {
        const int n = idx % d_state, t = idx / d_state;
        const float c_t = csum[size_t(first + t) * n_head + h] - base;
        const float* row = xbc + size_t(first + t) * row_n + d_inner;
        b_w[size_t(h) * n_bw + idx] = __float2half(row[size_t(g) * d_state + n] * __expf(ah * (last - c_t)));
        c_s[size_t(h) * n_bw + idx] = row[size_t(n_group + g) * d_state + n] * __expf(ah * c_t);
    }
    const int n_m = len * len;
    if (idx < n_m) {
        const int t_out = idx % len, t_in = idx / len;
        half v = __float2half(0.0f);
        if (t_in <= t_out) {
            const float co = csum[size_t(first + t_out) * n_head + h] - base;
            const float ci = csum[size_t(first + t_in) * n_head + h] - base;
            v = __float2half(__expf(ah * (co - ci)) * cb[size_t(g) * n_m + idx]);
        }
        m[size_t(h) * n_m + idx] = v;
    }
}

// S *= the decay over the chunk, before S += B' X_dt^T (ssm_ssd_scale_state_kernel)
__global__ void ssd_decay_kernel(float* __restrict__ state, const float* __restrict__ csum,
                                 const float* __restrict__ a, int per_head, int n_head, int first, int len) {
    const int h = blockIdx.y, idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= per_head) return;
    const float base = first > 0 ? csum[size_t(first - 1) * n_head + h] : 0.0f;
    const float last = csum[size_t(first + len - 1) * n_head + h] - base;
    state[size_t(h) * per_head + idx] *= __expf(a[h] * last);
}

// y += D x (build_mamba2_layer's D term, after the scan)
__global__ void ssd_d_kernel(float* __restrict__ y, const float* __restrict__ xbc, size_t row_n,
                             const float* __restrict__ d, int head_dim, int d_inner, int row0) {
    const size_t row = size_t(row0) + blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d_inner) return;
    y[row * d_inner + c] += xbc[row * row_n + c] * d[c / head_dim];
}

// Where the pieces of the scratch lie (floats, each 16-byte aligned); the halves take half as many floats.
struct SsdLayout {
    size_t dt_sp = 0, csum = 0, cb = 0, c_s = 0, x_dt = 0, b_w = 0, m = 0, total = 0;
};

SsdLayout ssd_layout(int rows, int n_head, int head_dim, int d_state, int n_group) {
    SsdLayout L;
    size_t at = 0;
    auto take = [&](size_t floats) {
        const size_t p = at;
        at += (floats + 3) & ~size_t(3);
        return p;
    };
    const size_t C = SSD_CHUNK;
    L.dt_sp = take(size_t(rows) * n_head);
    L.csum = take(size_t(rows) * n_head);
    L.cb = take(C * C * n_group);
    L.c_s = take(size_t(d_state) * C * n_head);
    L.x_dt = take((C * head_dim * n_head + 1) / 2);
    L.b_w = take((size_t(d_state) * C * n_head + 1) / 2);
    L.m = take((C * C * n_head + 1) / 2);
    L.total = at;
    return L;
}

void blas_ok(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("mamba2_scan_ssd: ") + what + ": cuBLAS status " +
                                 std::to_string((int) st));
}
#endif  // !STRATA_USE_HIP

}  // namespace

void mamba2_conv(const float* in, size_t in_stride, float* state, const float* w, const float* bias, float* out,
                 int channels, int rows, void* stream, float* snap, size_t stride, unsigned snaps) {
    if (rows <= 0) return;
    if (rows >= CONV_ROWS && (!snap || !snaps)) {
        for (int r = 0; r < rows; r += MAX_GRID_Y)
            conv_rows_kernel<<<dim3(blocks(channels, 256), std::min(MAX_GRID_Y, rows - r)), 256, 0, cs(stream)>>>(
                in, in_stride, state, w, bias, out, channels, r);
        conv_state_kernel<<<blocks(channels, 256), 256, 0, cs(stream)>>>(in, in_stride, state, channels, rows);
        return;
    }
    conv_walk_kernel<<<blocks(channels, 256), 256, 0, cs(stream)>>>(in, in_stride, state, w, bias, out, channels,
                                                                     rows, snap, stride, snap ? snaps : 0u);
}

void mamba2_scan(const float* xbc, const float* dt, size_t dt_stride, const float* dt_bias, const float* a,
                 const float* d, float* state, float* y, int n_head, int head_dim, int d_state, int n_group, int rows,
                 void* stream, float* snap, size_t stride, unsigned snaps) {
    if (d_state % WARP != 0 || d_state > 256 || d_state <= 0)
        throw std::invalid_argument("mamba2_scan: d_state is a multiple of 32, at most 256");
    if (n_group <= 0 || n_head % n_group != 0)
        throw std::invalid_argument("mamba2_scan: the heads are a whole number a group");
    if (rows <= 0) return;
    const int grid = blocks(size_t(n_head) * head_dim * WARP, 256);
    const unsigned sn = snap ? snaps : 0u;
    switch (d_state / WARP) {
    case 1: scan_kernel<1><<<grid, 256, 0, cs(stream)>>>(xbc, dt, dt_stride, dt_bias, a, d, state, y, n_head, head_dim,
                                                         n_group, rows, snap, stride, sn); break;
    case 2: scan_kernel<2><<<grid, 256, 0, cs(stream)>>>(xbc, dt, dt_stride, dt_bias, a, d, state, y, n_head, head_dim,
                                                         n_group, rows, snap, stride, sn); break;
    case 4: scan_kernel<4><<<grid, 256, 0, cs(stream)>>>(xbc, dt, dt_stride, dt_bias, a, d, state, y, n_head, head_dim,
                                                         n_group, rows, snap, stride, sn); break;
    case 8: scan_kernel<8><<<grid, 256, 0, cs(stream)>>>(xbc, dt, dt_stride, dt_bias, a, d, state, y, n_head, head_dim,
                                                         n_group, rows, snap, stride, sn); break;
    default: throw std::invalid_argument("mamba2_scan: d_state is 32, 64, 128 or 256");
    }
}

void mamba2_out(float* y, const float* z, size_t z_stride, const float* norm, int d_inner, int n_group, int rows,
                float eps, void* stream) {
    if (n_group <= 0 || d_inner % n_group != 0)
        throw std::invalid_argument("mamba2_out: the values are a whole number a group");
    for (int r = 0; r < rows; r += MAX_GRID_Y)
        out_kernel<<<dim3(n_group, std::min(MAX_GRID_Y, rows - r)), 256, 0, cs(stream)>>>(y, z, z_stride, norm, d_inner,
                                                                                         d_inner / n_group, r, eps);
}

#if !defined(STRATA_USE_HIP)
size_t mamba2_ssd_floats(int rows, int n_head, int head_dim, int d_state, int n_group) {
    return ssd_layout(std::min(rows, SSD_MAX_ROWS), n_head, head_dim, d_state, n_group).total;
}

bool mamba2_ssd_fits(int rows, int head_dim) {
    if (rows <= SSD_MIN_ROWS || rows > SSD_MAX_ROWS || head_dim % 8 != 0) return false;   // cuBLAS: 16-byte rows
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return major * 10 + minor >= 75;   // Turing or later, as llama.cpp
}

void mamba2_scan_ssd(const float* xbc, const float* dt, size_t dt_stride, const float* dt_bias, const float* a,
                     const float* d, float* state, float* y, int n_head, int head_dim, int d_state, int n_group,
                     int rows, void* blas, float* scratch, size_t scratch_floats, void* stream) {
    if (n_group <= 0 || n_head % n_group != 0)
        throw std::invalid_argument("mamba2_scan_ssd: the heads are a whole number a group");
    if (!mamba2_ssd_fits(rows, head_dim))
        throw std::invalid_argument("mamba2_scan_ssd: not for this call (mamba2_ssd_fits)");
    const SsdLayout L = ssd_layout(rows, n_head, head_dim, d_state, n_group);
    if (!blas || !scratch || L.total > scratch_floats)
        throw std::invalid_argument("mamba2_scan_ssd: no cuBLAS or scratch");
    const cudaStream_t s = cs(stream);
    const cublasHandle_t hb = (cublasHandle_t) blas;
    const int d_inner = n_head * head_dim, per_head = d_state * head_dim;
    const size_t row_n = size_t(d_inner) + 2 * size_t(n_group) * d_state;
    float* dt_sp = scratch + L.dt_sp;
    float* csum = scratch + L.csum;
    float* cb = scratch + L.cb;
    float* c_s = scratch + L.c_s;
    half* x_dt = (half*) (scratch + L.x_dt);
    half* b_w = (half*) (scratch + L.b_w);
    half* m = (half*) (scratch + L.m);
    ssd_dt_kernel<SSD_DT_BLOCK, SSD_DT_ITEMS><<<n_head, SSD_DT_BLOCK, 0, s>>>(dt, dt_stride, dt_bias, dt_sp, csum,
                                                                              n_head, rows);
    const float one = 1.0f, zero = 0.0f;
    const float* B0 = xbc + d_inner;
    const float* C0 = xbc + d_inner + size_t(n_group) * d_state;
    for (int first = 0; first < rows; first += SSD_CHUNK) {
        const int len = std::min(SSD_CHUNK, rows - first);
        // C B^T of each group: cb[t_out + t_in * len]
        blas_ok(cublasGemmStridedBatchedEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, len, len, d_state, &one,
                                           C0 + size_t(first) * row_n, CUDA_R_32F, (int) row_n, d_state,
                                           B0 + size_t(first) * row_n, CUDA_R_32F, (int) row_n, d_state, &zero,
                                           cb, CUDA_R_32F, len, (long long) len * len, n_group,
                                           CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT), "C B^T");
        const int work = std::max({len * head_dim, d_state * len, len * len});
        ssd_pre_kernel<256><<<dim3(blocks(size_t(work), 256), n_head), 256, 0, s>>>(
            csum, dt_sp, a, xbc, row_n, d_inner, x_dt, b_w, c_s, cb, m, len, head_dim, n_head, n_group, d_state, first);
        float* yc = y + size_t(first) * d_inner;
        // y = S^T C' (the state's share), then y += X_dt M^T (the chunk's own tokens)
        blas_ok(cublasGemmStridedBatchedEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, head_dim, len, d_state, &one,
                                           state, CUDA_R_32F, d_state, per_head,
                                           c_s, CUDA_R_32F, d_state, (long long) d_state * len, &zero,
                                           yc, CUDA_R_32F, d_inner, head_dim, n_head,
                                           CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT), "S^T C'");
        blas_ok(cublasGemmStridedBatchedEx(hb, CUBLAS_OP_N, CUBLAS_OP_T, head_dim, len, len, &one,
                                           x_dt, CUDA_R_16F, head_dim, (long long) len * head_dim,
                                           m, CUDA_R_16F, len, (long long) len * len, &one,
                                           yc, CUDA_R_32F, d_inner, head_dim, n_head,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "X_dt M^T");
        // S = decay S + B' X_dt^T
        ssd_decay_kernel<<<dim3(blocks(size_t(per_head), 256), n_head), 256, 0, s>>>(state, csum, a, per_head, n_head,
                                                                                    first, len);
        blas_ok(cublasGemmStridedBatchedEx(hb, CUBLAS_OP_N, CUBLAS_OP_T, d_state, head_dim, len, &one,
                                           b_w, CUDA_R_16F, d_state, (long long) d_state * len,
                                           x_dt, CUDA_R_16F, head_dim, (long long) len * head_dim, &one,
                                           state, CUDA_R_32F, d_state, per_head, n_head,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "B' X_dt^T");
    }
    for (int r = 0; r < rows; r += MAX_GRID_Y)
        ssd_d_kernel<<<dim3(blocks(size_t(d_inner), 256), std::min(MAX_GRID_Y, rows - r)), 256, 0, s>>>(
            y, xbc, row_n, d, head_dim, d_inner, r);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("mamba2_scan_ssd: ") + cudaGetErrorString(e));
}
#else
size_t mamba2_ssd_floats(int, int, int, int, int) { return 0; }

bool mamba2_ssd_fits(int, int) { return false; }

void mamba2_scan_ssd(const float*, const float*, size_t, const float*, const float*, const float*, float*, float*, int,
                     int, int, int, int, void*, float*, size_t, void*) {
    throw std::logic_error("mamba2_scan_ssd: no SSD form in a HIP build");
}
#endif  // !STRATA_USE_HIP

}  // namespace strata::poly
