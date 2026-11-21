// src/poly/mamba2.cu - the Mamba-2 mixer's kernels (include/strata/poly/mamba2.hpp): what llama.cpp's
// ggml_ssm_conv, ggml_ssm_scan (Mamba-2's form: one decay a head), the gate and the grouped RMS norm compute, with
// the recurrent state kept on the card and snapshots for a window of guessed tokens.  Tested by mamba2_test.cu
// against a host copy of llama.cpp's arithmetic (ggml-cpu ops.cpp, 08c252e22).
#include "strata/poly/mamba2.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

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

}  // namespace strata::poly
