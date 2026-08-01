// Q8_0 GEMM kernel - fused dequant + matrix-vector multiply.
//
// Layout (per 34-byte block, 32 elements):
//   bytes  0..2  : ggml_half d    (FP16 scale)
//   bytes  2..34 : int8_t  qs[32] (signed 8-bit quant values)
//
// Dequant:
//   y[i] = d * qs[i]
//
// Two levels of fused dot product are provided:
//   - dot_q8_0_block_naive: scalar reference, no SIMD.
//   - dot_q8_0_block_neon:  ARM NEON SIMD (16 elements per iteration).
//
// Public entry point: matmul_q8_0(raw, x, out, n, k).

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #define FRN_NEON 1
  #include <arm_neon.h>
#endif

namespace flatrun_native {

// ---------------------------------------------------------------------------
// Naive scalar Q8_0 block dot product
// ---------------------------------------------------------------------------
//
// For a single Q8_0 block w (32 elements) against an F32 input x of
// length 32, return the partial dot product.

static inline float dot_q8_0_block_naive(
    const uint8_t* block,
    const float* x
) {
    uint16_t d_h;
    std::memcpy(&d_h, block + 0, 2);
    float d = fp16_to_fp32(d_h);

    float acc = 0.0f;
    for (int i = 0; i < 32; ++i) {
        // qs is signed int8. Reinterpret the unsigned byte as signed.
        int8_t q = static_cast<int8_t>(block[2 + i]);
        acc += d * static_cast<float>(q) * x[i];
    }
    return acc;
}

// ---------------------------------------------------------------------------
// NEON SIMD Q8_0 block dot product
// ---------------------------------------------------------------------------

#if defined(FRN_NEON)

static inline float dot_q8_0_block_neon(
    const uint8_t* block,
    const float* x
) {
    uint16_t d_h;
    std::memcpy(&d_h, block + 0, 2);
    float d_f = fp16_to_fp32(d_h);

    // Load 32 signed int8 quant values from block+2.
    // Two lanes of 16 int8 each (vld1q_s8).
    int8x16_t qs0 = vld1q_s8(reinterpret_cast<const int8_t*>(block + 2));
    int8x16_t qs1 = vld1q_s8(reinterpret_cast<const int8_t*>(block + 18));

    // Convert int8 -> int16 -> int32 -> float32. 16 elements per chunk.
    int16x8_t qs0_lo = vmovl_s8(vget_low_s8(qs0));
    int16x8_t qs0_hi = vmovl_s8(vget_high_s8(qs0));
    int16x8_t qs1_lo = vmovl_s8(vget_low_s8(qs1));
    int16x8_t qs1_hi = vmovl_s8(vget_high_s8(qs1));

    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs0_lo)));
    float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs0_lo)));
    float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs0_hi)));
    float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs0_hi)));
    float32x4_t f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs1_lo)));
    float32x4_t f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs1_lo)));
    float32x4_t f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs1_hi)));
    float32x4_t f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs1_hi)));

    float32x4_t d = vdupq_n_f32(d_f);

    // Multiply by scale d. The activations x are still F32.
    f0 = vmulq_f32(f0, d);
    f1 = vmulq_f32(f1, d);
    f2 = vmulq_f32(f2, d);
    f3 = vmulq_f32(f3, d);
    f4 = vmulq_f32(f4, d);
    f5 = vmulq_f32(f5, d);
    f6 = vmulq_f32(f6, d);
    f7 = vmulq_f32(f7, d);

    float32x4_t x0 = vld1q_f32(x + 0);
    float32x4_t x1 = vld1q_f32(x + 4);
    float32x4_t x2 = vld1q_f32(x + 8);
    float32x4_t x3 = vld1q_f32(x + 12);
    float32x4_t x4 = vld1q_f32(x + 16);
    float32x4_t x5 = vld1q_f32(x + 20);
    float32x4_t x6 = vld1q_f32(x + 24);
    float32x4_t x7 = vld1q_f32(x + 28);

    float32x4_t acc = vdupq_n_f32(0.0f);
    acc = vmlaq_f32(acc, f0, x0);
    acc = vmlaq_f32(acc, f1, x1);
    acc = vmlaq_f32(acc, f2, x2);
    acc = vmlaq_f32(acc, f3, x3);
    acc = vmlaq_f32(acc, f4, x4);
    acc = vmlaq_f32(acc, f5, x5);
    acc = vmlaq_f32(acc, f6, x6);
    acc = vmlaq_f32(acc, f7, x7);

    float32x2_t lo = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    float32x2_t pair = vpadd_f32(lo, lo);
    return vget_lane_f32(pair, 0);
}

static inline void dequant_q8_0_block_neon(
    const uint8_t* block,
    float* out
) {
    uint16_t d_h;
    std::memcpy(&d_h, block + 0, 2);
    float d_f = fp16_to_fp32(d_h);

    int8x16_t qs0 = vld1q_s8(reinterpret_cast<const int8_t*>(block + 2));
    int8x16_t qs1 = vld1q_s8(reinterpret_cast<const int8_t*>(block + 18));

    int16x8_t qs0_lo = vmovl_s8(vget_low_s8(qs0));
    int16x8_t qs0_hi = vmovl_s8(vget_high_s8(qs0));
    int16x8_t qs1_lo = vmovl_s8(vget_low_s8(qs1));
    int16x8_t qs1_hi = vmovl_s8(vget_high_s8(qs1));

    float32x4_t d = vdupq_n_f32(d_f);

    vst1q_f32(out + 0,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs0_lo))), d));
    vst1q_f32(out + 4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs0_lo))), d));
    vst1q_f32(out + 8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs0_hi))), d));
    vst1q_f32(out + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs0_hi))), d));
    vst1q_f32(out + 16, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs1_lo))), d));
    vst1q_f32(out + 20, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs1_lo))), d));
    vst1q_f32(out + 24, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs1_hi))), d));
    vst1q_f32(out + 28, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs1_hi))), d));
}

#endif  // FRN_NEON

// ---------------------------------------------------------------------------
// Q8_0 dequant of a whole tensor (validation path)
// ---------------------------------------------------------------------------

static inline void dequant_q8_0_naive(
    const uint8_t* raw,
    std::size_t n_blocks,
    float* out
) {
    for (std::size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* block = raw + b * 34;
        uint16_t d_h;
        std::memcpy(&d_h, block + 0, 2);
        float d = fp16_to_fp32(d_h);
        for (int i = 0; i < 32; ++i) {
            int8_t q = static_cast<int8_t>(block[2 + i]);
            out[b * 32 + i] = d * static_cast<float>(q);
        }
    }
}

// ---------------------------------------------------------------------------
// Q8_0 x FP32 matrix-vector multiply
// ---------------------------------------------------------------------------
//
// weight: shape (n, k) packed Q8_0 (n rows, k cols of in_features).
//         k must be a multiple of 32; weight byte size = n * (k/32) * 34.
// x:      shape (k,) FP32 input activations.
// out:    shape (n,) FP32 output: out[i] = sum_j w[i,j] * x[j]

#if defined(FRN_NEON)
typedef float (*dot_block_q8_0_fn)(const uint8_t*, const float*);
#endif

static inline void matmul_q8_0(
    const uint8_t* weight,
    const float* x,
    float* out,
    std::size_t n,
    std::size_t k
) {
    std::size_t n_blocks = k / 32;
#if defined(FRN_NEON)
    dot_block_q8_0_fn dot_block = dot_q8_0_block_neon;
#else
    auto dot_block = dot_q8_0_block_naive;
#endif
    for (std::size_t i = 0; i < n; ++i) {
        const uint8_t* row = weight + i * n_blocks * 34;
        float acc = 0.0f;
        for (std::size_t b = 0; b < n_blocks; ++b) {
            acc += dot_block(row + b * 34, x + b * 32);
        }
        out[i] = acc;
    }
}

#if defined(FRN_NEON)
#include <thread>
#include <vector>

// Multi-threaded Q8_0 matmul: parallel across rows. See matmul_q4_k_mt
// in q4_k.h for the design notes.
static inline void matmul_q8_0_mt(
    const uint8_t* weight,
    const float* x,
    float* out,
    std::size_t n,
    std::size_t k,
    int n_threads = 0
) {
    std::size_t n_blocks = k / 32;
    dot_block_q8_0_fn dot_block = dot_q8_0_block_neon;
    if (n_threads <= 0) {
        n_threads = (int)std::thread::hardware_concurrency();
        if (n_threads < 1) n_threads = 1;
    }
    if ((std::size_t)n_threads > n) n_threads = (int)n;
    if (n_threads <= 1) {
        for (std::size_t i = 0; i < n; ++i) {
            const uint8_t* row = weight + i * n_blocks * 34;
            float acc = 0.0f;
            for (std::size_t b = 0; b < n_blocks; ++b) {
                acc += dot_block(row + b * 34, x + b * 32);
            }
            out[i] = acc;
        }
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    auto worker = [&](std::size_t start, std::size_t end) {
        for (std::size_t i = start; i < end; ++i) {
            const uint8_t* row = weight + i * n_blocks * 34;
            float acc = 0.0f;
            for (std::size_t b = 0; b < n_blocks; ++b) {
                acc += dot_block(row + b * 34, x + b * 32);
            }
            out[i] = acc;
        }
    };
    std::size_t chunk = (n + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; ++t) {
        std::size_t start = (std::size_t)t * chunk;
        std::size_t end = std::min(start + chunk, n);
        if (start >= end) break;
        workers.emplace_back(worker, start, end);
    }
    for (auto& w : workers) w.join();
}
#endif

}  // namespace flatrun_native
