// src/poly/mamba2_test.cu - the Mamba-2 kernels (strata/poly/mamba2.hpp) against a host copy of llama.cpp's
// arithmetic for the layer (ggml-cpu ops.cpp ssm_conv / ssm_scan at 08c252e22, src/models/mamba-base.cpp
// build_mamba2_layer).  Random inputs, no model file:
//   1. the convolution: a walk of 5 tokens, then 20 at once (its other kernel), the state carried between the calls;
//      the outputs and the state after them; then a walk that leaves snapshots after tokens 0, 2 and 4;
//   2. the scan: 5 tokens, then 20, dt read with a stride as in ssm_in's rows; y and the state; snapshots after
//      tokens 1 and 3;
//   3. the gate and the grouped RMS norm;
//   4. the scan at Nemotron 3.5 Bolt's sizes (128 heads of 64 values, d_state 128, 8 groups), 3 tokens.
// usage: poly_mamba2_test   (exit 0: every case agrees; 1: a mismatch, printed; 77: no card)
#include "strata/poly/mamba2.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using strata::poly::mamba2_conv;
using strata::poly::mamba2_out;
using strata::poly::mamba2_scan;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "poly_mamba2_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Buf {
    float* p = nullptr;
    size_t n = 0;
    explicit Buf(size_t count) : n(count) { ck(cudaMalloc(&p, (count ? count : 1) * sizeof(float)), "cudaMalloc"); }
    explicit Buf(const std::vector<float>& v) : Buf(v.size()) { put(v); }
    ~Buf() { cudaFree(p); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    void put(const std::vector<float>& v) { ck(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "put"); }
    std::vector<float> get() const {
        std::vector<float> v(n);
        ck(cudaDeviceSynchronize(), "kernel");
        ck(cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost), "get");
        return v;
    }
};

std::vector<float> rnd(std::mt19937& g, size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (float& x : v) x = u(g);
    return v;
}

int failures = 0;

// Every value within 1e-4 of the reference, relative to its size (at least 1).
void same(const char* what, const std::vector<float>& got, const std::vector<float>& want, size_t n = 0) {
    if (!n) n = want.size();
    double worst = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs((double) got[i] - want[i]) / std::max(1.0, std::fabs((double) want[i]));
        if (!(e <= worst)) worst = e, at = i;   // NaN counts as the worst
    }
    const bool ok = worst <= 1e-4;
    std::printf("  %-44s %s (worst %.2e at %zu: %.7g vs %.7g)\n", what, ok ? "ok" : "MISMATCH", worst, at,
                (double) got[at], (double) want[at]);
    if (!ok) ++failures;
}

// ---- the host reference: llama.cpp's order of operations
float silu(float x) { return x / (1.0f + std::exp(-x)); }
float softplus(float x) { return x > 20.0f ? x : std::log(1.0f + std::exp(x)); }

// ggml_ssm_conv over the 3 past inputs and the tokens, + bias, silu; the state advances.
void ref_conv(const std::vector<float>& in, size_t in_stride, std::vector<float>& state, const std::vector<float>& w,
              const std::vector<float>& bias, std::vector<float>& out, int channels, int rows, int row0 = 0,
              std::vector<std::vector<float>>* states_after = nullptr) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < channels; ++c) {
            float* st = &state[(size_t) c * 3];
            const float v = in[(size_t) r * in_stride + c];
            float sumf = 0.0f;
            sumf += st[0] * w[(size_t) c * 4 + 0];
            sumf += st[1] * w[(size_t) c * 4 + 1];
            sumf += st[2] * w[(size_t) c * 4 + 2];
            sumf += v * w[(size_t) c * 4 + 3];
            out[(size_t) (row0 + r) * channels + c] = silu(sumf + bias[c]);
            st[0] = st[1];
            st[1] = st[2];
            st[2] = v;
        }
        if (states_after) states_after->push_back(state);
    }
}

// ggml_ssm_scan (Mamba-2: a scalar A a head) and y + D * x.
void ref_scan(const std::vector<float>& xbc, const std::vector<float>& dt, size_t dt_stride,
              const std::vector<float>& dt_b, const std::vector<float>& a, const std::vector<float>& d,
              std::vector<float>& state, std::vector<float>& y, int nh, int hd, int ds, int ng, int rows, int row0 = 0,
              std::vector<std::vector<float>>* states_after = nullptr) {
    const size_t d_inner = (size_t) nh * hd, row_n = d_inner + 2 * (size_t) ng * ds;
    for (int r = 0; r < rows; ++r) {
        const float* x = &xbc[(size_t) r * row_n];
        const float* B = x + d_inner;
        const float* C = B + (size_t) ng * ds;
        for (int h = 0; h < nh; ++h) {
            const float dsp = softplus(dt[(size_t) r * dt_stride + h] + dt_b[h]);
            const float dA = std::exp(dsp * a[h]);
            const int g = h / (nh / ng);
            for (int i = 0; i < hd; ++i) {
                const size_t ii = (size_t) h * hd + i;
                const float x_dt = x[ii] * dsp;
                float sumf = 0.0f;
                for (int s = 0; s < ds; ++s) {
                    float& st = state[ii * ds + s];
                    st = st * dA + B[(size_t) g * ds + s] * x_dt;
                    sumf += st * C[(size_t) g * ds + s];
                }
                y[(size_t) (row0 + r) * d_inner + ii] = sumf + x[ii] * d[h];
            }
        }
        if (states_after) states_after->push_back(state);
    }
}

void ref_out(std::vector<float>& y, const std::vector<float>& z, size_t z_stride, const std::vector<float>& norm,
             int d_inner, int ng, int rows, float eps) {
    const int per = d_inner / ng;
    for (int r = 0; r < rows; ++r)
        for (int g = 0; g < ng; ++g) {
            float* yr = &y[(size_t) r * d_inner + (size_t) g * per];
            const float* zr = &z[(size_t) r * z_stride + (size_t) g * per];
            double sum = 0.0;
            for (int c = 0; c < per; ++c) {
                yr[c] = silu(zr[c]) * yr[c];
                sum += (double) yr[c] * yr[c];
            }
            const float scale = 1.0f / std::sqrt((float) (sum / per) + eps);
            for (int c = 0; c < per; ++c) yr[c] = yr[c] * scale * norm[(size_t) g * per + c];
        }
}

// ---- the cases
void conv_case(std::mt19937& g) {
    std::puts("convolution (1536 channels, 4 taps, bias)");
    const int ch = 1536, a_rows = 5, b_rows = 20, rows = a_rows + b_rows;
    const size_t stride = ch + 77;   // its part of a wider row, as xBC in ssm_in's
    const auto in = rnd(g, (size_t) rows * stride, -2.f, 2.f), w = rnd(g, (size_t) ch * 4, -1.f, 1.f),
               bias = rnd(g, ch, -0.5f, 0.5f), st0 = rnd(g, (size_t) ch * 3, -2.f, 2.f);
    std::vector<float> want((size_t) rows * ch), st = st0;
    ref_conv(in, stride, st, w, bias, want, ch, rows);
    Buf din(in), dw(w), db(bias), dst(st0), dout((size_t) rows * ch);
    mamba2_conv(din.p, stride, dst.p, dw.p, db.p, dout.p, ch, a_rows, nullptr);                    // the walk
    mamba2_conv(din.p + a_rows * stride, stride, dst.p, dw.p, db.p, dout.p + (size_t) a_rows * ch, ch, b_rows,
                nullptr);                                                                              // at once
    same("5 tokens walked, then 20 at once", dout.get(), want);
    same("the state after them", dst.get(), st);

    // snapshots after tokens 0, 2 and 4 of a 5-token walk
    std::vector<std::vector<float>> after;
    std::vector<float> st2 = st0, want2((size_t) a_rows * ch);
    ref_conv(in, stride, st2, w, bias, want2, ch, a_rows, 0, &after);
    Buf dst2(st0), dout2((size_t) a_rows * ch), snap((size_t) 3 * ch * 3);
    mamba2_conv(din.p, stride, dst2.p, dw.p, db.p, dout2.p, ch, a_rows, nullptr, snap.p, (size_t) ch * 3, 0b10101u);
    same("a walk that leaves snapshots", dout2.get(), want2);
    const auto s = snap.get();
    for (int k = 0; k < 3; ++k) {
        char what[64];
        std::snprintf(what, sizeof what, "its snapshot after token %d", 2 * k);
        same(what, std::vector<float>(s.begin() + (size_t) k * ch * 3, s.begin() + (size_t) (k + 1) * ch * 3),
             after[(size_t) 2 * k]);
    }
}

void scan_case(std::mt19937& g, int nh, int hd, int ds, int ng, int a_rows, int b_rows, bool snaps) {
    std::printf("scan (%d heads of %d values, d_state %d, %d groups, %d + %d tokens)\n", nh, hd, ds, ng, a_rows, b_rows);
    const int rows = a_rows + b_rows;
    const size_t d_inner = (size_t) nh * hd, row_n = d_inner + 2 * (size_t) ng * ds, dt_stride = nh + 13;
    const auto xbc = rnd(g, (size_t) rows * row_n, -1.f, 1.f), dt = rnd(g, (size_t) rows * dt_stride, -3.f, 1.f),
               dt_b = rnd(g, nh, -1.f, 1.f), d = rnd(g, nh, -1.f, 1.f), st0 = rnd(g, d_inner * ds, -1.f, 1.f);
    std::vector<float> a = rnd(g, nh, -1.f, 2.f);
    for (float& x : a) x = -std::exp(x);   // stored as -exp(A_log)
    std::vector<float> want((size_t) rows * d_inner), st = st0;
    ref_scan(xbc, dt, dt_stride, dt_b, a, d, st, want, nh, hd, ds, ng, rows);
    Buf dx(xbc), ddt(dt), ddb(dt_b), da(a), dd(d), dst(st0), dy((size_t) rows * d_inner);
    mamba2_scan(dx.p, ddt.p, dt_stride, ddb.p, da.p, dd.p, dst.p, dy.p, nh, hd, ds, ng, a_rows, nullptr);
    if (b_rows)
        mamba2_scan(dx.p + a_rows * row_n, ddt.p + a_rows * dt_stride, dt_stride, ddb.p, da.p, dd.p, dst.p,
                    dy.p + a_rows * d_inner, nh, hd, ds, ng, b_rows, nullptr);
    same("y", dy.get(), want);
    same("the state after them", dst.get(), st);
    if (!snaps) return;

    std::vector<std::vector<float>> after;
    std::vector<float> st2 = st0, want2((size_t) a_rows * d_inner);
    ref_scan(xbc, dt, dt_stride, dt_b, a, d, st2, want2, nh, hd, ds, ng, a_rows, 0, &after);
    const size_t state_n = d_inner * ds;
    Buf dst2(st0), dy2((size_t) a_rows * d_inner), snap(2 * state_n);
    mamba2_scan(dx.p, ddt.p, dt_stride, ddb.p, da.p, dd.p, dst2.p, dy2.p, nh, hd, ds, ng, a_rows, nullptr, snap.p,
                state_n, 0b01010u);
    same("y with snapshots", dy2.get(), want2);
    const auto s = snap.get();
    same("its snapshot after token 1", std::vector<float>(s.begin(), s.begin() + state_n), after[1]);
    same("its snapshot after token 3", std::vector<float>(s.begin() + state_n, s.end()), after[3]);
}

void out_case(std::mt19937& g) {
    std::puts("gate and grouped RMS norm (1024 values, 2 groups, 7 tokens)");
    const int d_inner = 1024, ng = 2, rows = 7;
    const size_t z_stride = 2 * d_inner + 300;
    const float eps = 1e-5f;
    auto y = rnd(g, (size_t) rows * d_inner, -3.f, 3.f);
    const auto z = rnd(g, (size_t) rows * z_stride, -3.f, 3.f), norm = rnd(g, d_inner, 0.f, 2.f);
    Buf dy(y), dz(z), dn(norm);
    mamba2_out(dy.p, dz.p, z_stride, dn.p, d_inner, ng, rows, eps, nullptr);
    ref_out(y, z, z_stride, norm, d_inner, ng, rows, eps);
    same("y", dy.get(), y);
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("poly_mamba2_test: SKIP (no card)");
        return 77;
    }
    std::mt19937 g(20261121);
    conv_case(g);
    scan_case(g, 16, 64, 128, 2, 5, 20, true);
    out_case(g);
    scan_case(g, 128, 64, 128, 8, 3, 0, false);   // Bolt's sizes
    if (failures) {
        std::printf("poly_mamba2_test: FAIL (%d)\n", failures);
        return 1;
    }
    std::puts("poly_mamba2_test: PASS");
    return 0;
}
