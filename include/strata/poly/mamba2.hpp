// include/strata/poly/mamba2.hpp - the Mamba-2 mixer's kernels for strata-poly's Nemotron-H models
// (`nemotron_h_moe`: Nemotron 3.5 Bolt, Super).  They compute what llama.cpp computes for the layer
// (src/models/mamba-base.cpp build_mamba2_layer: ggml_ssm_conv, ggml_ssm_scan, the gate and the grouped RMS norm):
//
//   zxBCdt = ssm_in(x)                                        a token's row: z | xBC | dt
//   xBC    = silu(conv4(xBC, state) + conv_bias)               mamba2_conv
//   y      = scan(x, B, C, dt, A, state) + D * x               mamba2_scan (x, B, C: the parts of xBC)
//   y      = rms_norm_groups(silu(z) * y) * norm               mamba2_out
//   out    = ssm_out(y)
//
// CUDA and HIP.  The recurrent state is the layer's: the conv's 3 past inputs a channel, and the scan's d_state
// values for each value of each head.  A window of guessed tokens can keep the states as they are after some of its
// tokens (snap / stride / snaps, as conv4_silu and gdn_step do): `snaps` has a bit for each of the call's tokens that
// leaves one, and those states lie in the tokens' order, `stride` floats apart, each at the layer's own place.
#pragma once

#include <cstddef>

namespace strata::poly {

/// The causal convolution of four taps over each of `channels` channels, its bias and its SiLU, token after token.
/// in: a token's `channels` inputs at in + row * in_stride (the xBC part of ssm_in's row).  state: 3 past inputs a
/// channel, oldest first, advanced by `rows` tokens.  w: (4, channels).  bias: `channels` (may be null).
/// out: `channels` values a token, back to back.
void mamba2_conv(const float* in, size_t in_stride, float* state, const float* w, const float* bias, float* out,
                 int channels, int rows, void* stream, float* snap = nullptr, size_t stride = 0, unsigned snaps = 0);

/// The selective scan of every head, token after token.  xbc: mamba2_conv's output, n_head * head_dim x values,
/// then n_group * d_state B values, then n_group * d_state C values a token; head h reads group h / (n_head /
/// n_group).  dt: n_head values a token at dt + row * dt_stride (ssm_in's dt part), dt_bias added, then softplus.
/// a: n_head (stored negative), d: n_head.  state: n_head * head_dim * d_state, the value i of head h at
/// (h * head_dim + i) * d_state, updated.  y: n_head * head_dim values a token, back to back: C . state + D * x.
/// d_state: a multiple of 32, at most 256.
void mamba2_scan(const float* xbc, const float* dt, size_t dt_stride, const float* dt_bias, const float* a,
                 const float* d, float* state, float* y, int n_head, int head_dim, int d_state, int n_group, int rows,
                 void* stream, float* snap = nullptr, size_t stride = 0, unsigned snaps = 0);

/// The gate and the grouped RMS norm, in place: y = rms_norm(silu(z) * y) * norm over each of the n_group groups
/// of d_inner / n_group values.  y: d_inner values a token, back to back.  z: d_inner values a token at
/// z + row * z_stride (ssm_in's z part).  norm: d_inner (group after group).  d_inner / n_group: a multiple of 32.
void mamba2_out(float* y, const float* z, size_t z_stride, const float* norm, int d_inner, int n_group, int rows,
                float eps, void* stream);

}  // namespace strata::poly
