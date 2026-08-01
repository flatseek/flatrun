// Q4_K GEMM kernel — fused dequant + matrix-vector multiply.
//
// Layout (per 144-byte block, 256 elements):
//   bytes  0..2   : ggml_half d       (FP16 scale)
//   bytes  2..4   : ggml_half dmin    (FP16 min scale)
//   bytes  4..16  : uint8_t scales[12]   (8 pairs of 6-bit scale/min)
//   bytes 16..144 : uint8_t qs[128]      (4 groups of 32 bytes packed nibbles)
//
// Per-group dequant (4 groups of 64 elements each):
//   for g in 0..4:
//     lo = qs[g*32 : g*32+32] & 0x0F      -> elements [g*64+0  .. g*64+31]
//     hi = qs[g*32 : g*32+32] >> 4         -> elements [g*64+32 .. g*64+63]
//     y[g*64+0..32]  = d * sc[2g]   * lo - dmin * m[2g]
//     y[g*64+32..64] = d * sc[2g+1] * hi - dmin * m[2g+1]
//
// The 6-bit scale/min packing is the same as ggml-quants.c: for j<4 the
// value sits in the low 6 bits of q[j] / q[j+4]; for j>=4 the value
// is split across bytes (low 4 bits + 2 high bits from an earlier byte).
//
// We expose:
//   - dequant_q4_k_block_naive: scalar reference, no SIMD.
//   - dequant_q4_k_block_neon:  ARM NEON SIMD (4 quantities per inner loop).
//   - dot_q4_k_block_naive:  scalar fused dequant + dot product.
//   - dot_q4_k_block_neon:   ARM NEON SIMD fused dequant + dot product.
//   - matmul_q4_k: high-level dispatch.

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------
// Architecture detection
// ---------------------------------------------------------------------------

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #define FRN_NEON 1
  #include <arm_neon.h>
#endif

namespace flatrun_native {

// ---------------------------------------------------------------------------
// Half-precision conversion (IEEE 754 binary16 -> float32)
// ---------------------------------------------------------------------------
//
// ggml stores the FP16 scale/dmin at the head of every Q4_K block in
// little-endian wire format. We convert without relying on
// `_cvtsh_ss` (which is x86) or a runtime helper - just bit-level
// composition. The behaviour is well-defined for the 5 subnormals
// and quiet/payload NaNs the model loading path can produce.

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (uint32_t)(h & 0x7C00) >> 10;
    uint32_t mant = (uint32_t)(h & 0x03FF);
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                shift++;
            }
            mant &= 0x03FF;
            bits = sign | ((uint32_t)(127 - 15 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// ---------------------------------------------------------------------------
// 6-bit scale/min unpack (8 pairs)
// ---------------------------------------------------------------------------
//
// Input: 12 bytes (uint8). Output: 8 floats each for sc and m.
// Same algorithm as ggml-quants.c::get_scale_min_k4.

static inline void unpack_scale_min_k4(
    const uint8_t* q,
    float sc[8],
    float m[8]
) {
    sc[0] = (float)(q[0] & 0x3F);
    sc[1] = (float)(q[1] & 0x3F);
    sc[2] = (float)(q[2] & 0x3F);
    sc[3] = (float)(q[3] & 0x3F);
    m[0]  = (float)(q[4] & 0x3F);
    m[1]  = (float)(q[5] & 0x3F);
    m[2]  = (float)(q[6] & 0x3F);
    m[3]  = (float)(q[7] & 0x3F);
    sc[4] = (float)((q[8]  & 0x0F) | ((q[0] >> 6) << 4));
    sc[5] = (float)((q[9]  & 0x0F) | ((q[1] >> 6) << 4));
    sc[6] = (float)((q[10] & 0x0F) | ((q[2] >> 6) << 4));
    sc[7] = (float)((q[11] & 0x0F) | ((q[3] >> 6) << 4));
    m[4]  = (float)((q[8]  >> 4) | ((q[4] >> 6) << 4));
    m[5]  = (float)((q[9]  >> 4) | ((q[5] >> 6) << 4));
    m[6]  = (float)((q[10] >> 4) | ((q[6] >> 6) << 4));
    m[7]  = (float)((q[11] >> 4) | ((q[7] >> 6) << 4));
}

// ---------------------------------------------------------------------------
// Naive scalar Q4_K block dequant (256 elements -> FP32)
// ---------------------------------------------------------------------------

static inline void dequant_q4_k_block_naive(
    const uint8_t* block,
    float* out
) {
    uint16_t d_h, dmin_h;
    std::memcpy(&d_h,    block + 0, 2);
    std::memcpy(&dmin_h, block + 2, 2);
    float d    = fp16_to_fp32(d_h);
    float dmin = fp16_to_fp32(dmin_h);

    float sc[8], m[8];
    unpack_scale_min_k4(block + 4, sc, m);

    const uint8_t* qs = block + 16;
    for (int g = 0; g < 4; ++g) {
        const uint8_t* chunk = qs + g * 32;
        int lo_off = g * 64;
        int hi_off = g * 64 + 32;
        float s0 = d * sc[2 * g];
        float s1 = d * sc[2 * g + 1];
        float t0 = dmin * m[2 * g];
        float t1 = dmin * m[2 * g + 1];
        for (int i = 0; i < 32; ++i) {
            uint8_t qb = chunk[i];
            out[lo_off + i]      = s0 * (float)(qb & 0x0F) - t0;
            out[hi_off + i]      = s1 * (float)(qb >> 4)    - t1;
        }
    }
}

// ---------------------------------------------------------------------------
// Fused dequant + dot product: out = sum_i w_i * x_i
// ---------------------------------------------------------------------------
//
// For a single Q4_K block w (256 elements) against an F32 input x of
// length 256, the partial dot product is computed during dequant.

static inline float dot_q4_k_block_naive(
    const uint8_t* block,
    const float* x
) {
    uint16_t d_h, dmin_h;
    std::memcpy(&d_h,    block + 0, 2);
    std::memcpy(&dmin_h, block + 2, 2);
    float d    = fp16_to_fp32(d_h);
    float dmin = fp16_to_fp32(dmin_h);

    float sc[8], m[8];
    unpack_scale_min_k4(block + 4, sc, m);

    const uint8_t* qs = block + 16;
    float acc = 0.0f;
    for (int g = 0; g < 4; ++g) {
        const uint8_t* chunk = qs + g * 32;
        int lo_off = g * 64;
        int hi_off = g * 64 + 32;
        float s0 = d * sc[2 * g];
        float s1 = d * sc[2 * g + 1];
        float t0 = dmin * m[2 * g];
        float t1 = dmin * m[2 * g + 1];
        for (int i = 0; i < 32; ++i) {
            uint8_t qb = chunk[i];
            float ql = (float)(qb & 0x0F);
            float qh = (float)(qb >> 4);
            acc += (s0 * ql - t0) * x[lo_off + i];
            acc += (s1 * qh - t1) * x[hi_off + i];
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// NEON SIMD Q4_K block dot product
// ---------------------------------------------------------------------------

#if defined(FRN_NEON)

static inline float dot_q4_k_block_neon(
    const uint8_t* block,
    const float* x
) {
    uint16_t d_h, dmin_h;
    std::memcpy(&d_h,    block + 0, 2);
    std::memcpy(&dmin_h, block + 2, 2);
    float d_f    = fp16_to_fp32(d_h);
    float dmin_f = fp16_to_fp32(dmin_h);

    float sc[8], m[8];
    unpack_scale_min_k4(block + 4, sc, m);

    const uint8_t* qs = block + 16;
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (int g = 0; g < 4; ++g) {
        const uint8_t* chunk = qs + g * 32;
        uint8x16_t q0 = vld1q_u8(chunk + 0);
        uint8x16_t q1 = vld1q_u8(chunk + 16);

        uint8x16_t lo0 = vandq_u8(q0, vdupq_n_u8(0x0F));
        uint8x16_t hi0 = vshrq_n_u8(q0, 4);
        uint8x16_t lo1 = vandq_u8(q1, vdupq_n_u8(0x0F));
        uint8x16_t hi1 = vshrq_n_u8(q1, 4);

        uint16x8_t lo_w0 = vmovl_u8(vget_low_u8(lo0));
        uint16x8_t lo_w1 = vmovl_u8(vget_high_u8(lo0));
        uint16x8_t lo_w2 = vmovl_u8(vget_low_u8(lo1));
        uint16x8_t lo_w3 = vmovl_u8(vget_high_u8(lo1));

        float32x4_t lo_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_w0)));
        float32x4_t lo_f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_w0)));
        float32x4_t lo_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_w1)));
        float32x4_t lo_f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_w1)));
        float32x4_t lo_f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_w2)));
        float32x4_t lo_f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_w2)));
        float32x4_t lo_f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_w3)));
        float32x4_t lo_f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_w3)));

        uint16x8_t hi_w0 = vmovl_u8(vget_low_u8(hi0));
        uint16x8_t hi_w1 = vmovl_u8(vget_high_u8(hi0));
        uint16x8_t hi_w2 = vmovl_u8(vget_low_u8(hi1));
        uint16x8_t hi_w3 = vmovl_u8(vget_high_u8(hi1));

        float32x4_t hi_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_w0)));
        float32x4_t hi_f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_w0)));
        float32x4_t hi_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_w1)));
        float32x4_t hi_f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_w1)));
        float32x4_t hi_f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_w2)));
        float32x4_t hi_f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_w2)));
        float32x4_t hi_f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_w3)));
        float32x4_t hi_f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_w3)));

        float32x4_t s0 = vdupq_n_f32(d_f * sc[2 * g]);
        float32x4_t s1 = vdupq_n_f32(d_f * sc[2 * g + 1]);
        float32x4_t neg_t0 = vdupq_n_f32(-(dmin_f * m[2 * g]));
        float32x4_t neg_t1 = vdupq_n_f32(-(dmin_f * m[2 * g + 1]));

        // dequant q*v - dmin*m = vmlaq_f32(neg_t, s, q) - neg_t is the
        // negated constant, so the result is -dmin*m + s*q = s*q - dmin*m.
        float32x4_t lo_v0 = vmlaq_f32(neg_t0, s0, lo_f0);
        float32x4_t lo_v1 = vmlaq_f32(neg_t0, s0, lo_f1);
        float32x4_t lo_v2 = vmlaq_f32(neg_t0, s0, lo_f2);
        float32x4_t lo_v3 = vmlaq_f32(neg_t0, s0, lo_f3);
        float32x4_t lo_v4 = vmlaq_f32(neg_t0, s0, lo_f4);
        float32x4_t lo_v5 = vmlaq_f32(neg_t0, s0, lo_f5);
        float32x4_t lo_v6 = vmlaq_f32(neg_t0, s0, lo_f6);
        float32x4_t lo_v7 = vmlaq_f32(neg_t0, s0, lo_f7);

        float32x4_t hi_v0 = vmlaq_f32(neg_t1, s1, hi_f0);
        float32x4_t hi_v1 = vmlaq_f32(neg_t1, s1, hi_f1);
        float32x4_t hi_v2 = vmlaq_f32(neg_t1, s1, hi_f2);
        float32x4_t hi_v3 = vmlaq_f32(neg_t1, s1, hi_f3);
        float32x4_t hi_v4 = vmlaq_f32(neg_t1, s1, hi_f4);
        float32x4_t hi_v5 = vmlaq_f32(neg_t1, s1, hi_f5);
        float32x4_t hi_v6 = vmlaq_f32(neg_t1, s1, hi_f6);
        float32x4_t hi_v7 = vmlaq_f32(neg_t1, s1, hi_f7);

        int lo_off = g * 64;
        int hi_off = g * 64 + 32;
        float32x4_t x_lo0 = vld1q_f32(x + lo_off + 0);
        float32x4_t x_lo1 = vld1q_f32(x + lo_off + 4);
        float32x4_t x_lo2 = vld1q_f32(x + lo_off + 8);
        float32x4_t x_lo3 = vld1q_f32(x + lo_off + 12);
        float32x4_t x_lo4 = vld1q_f32(x + lo_off + 16);
        float32x4_t x_lo5 = vld1q_f32(x + lo_off + 20);
        float32x4_t x_lo6 = vld1q_f32(x + lo_off + 24);
        float32x4_t x_lo7 = vld1q_f32(x + lo_off + 28);

        float32x4_t x_hi0 = vld1q_f32(x + hi_off + 0);
        float32x4_t x_hi1 = vld1q_f32(x + hi_off + 4);
        float32x4_t x_hi2 = vld1q_f32(x + hi_off + 8);
        float32x4_t x_hi3 = vld1q_f32(x + hi_off + 12);
        float32x4_t x_hi4 = vld1q_f32(x + hi_off + 16);
        float32x4_t x_hi5 = vld1q_f32(x + hi_off + 20);
        float32x4_t x_hi6 = vld1q_f32(x + hi_off + 24);
        float32x4_t x_hi7 = vld1q_f32(x + hi_off + 28);

        acc = vmlaq_f32(acc, lo_v0, x_lo0);
        acc = vmlaq_f32(acc, lo_v1, x_lo1);
        acc = vmlaq_f32(acc, lo_v2, x_lo2);
        acc = vmlaq_f32(acc, lo_v3, x_lo3);
        acc = vmlaq_f32(acc, lo_v4, x_lo4);
        acc = vmlaq_f32(acc, lo_v5, x_lo5);
        acc = vmlaq_f32(acc, lo_v6, x_lo6);
        acc = vmlaq_f32(acc, lo_v7, x_lo7);

        acc = vmlaq_f32(acc, hi_v0, x_hi0);
        acc = vmlaq_f32(acc, hi_v1, x_hi1);
        acc = vmlaq_f32(acc, hi_v2, x_hi2);
        acc = vmlaq_f32(acc, hi_v3, x_hi3);
        acc = vmlaq_f32(acc, hi_v4, x_hi4);
        acc = vmlaq_f32(acc, hi_v5, x_hi5);
        acc = vmlaq_f32(acc, hi_v6, x_hi6);
        acc = vmlaq_f32(acc, hi_v7, x_hi7);
    }

    float32x2_t lo = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    float32x2_t pair = vpadd_f32(lo, lo);
    return vget_lane_f32(pair, 0);
}

static inline void dequant_q4_k_block_neon(
    const uint8_t* block,
    float* out
) {
    uint16_t d_h, dmin_h;
    std::memcpy(&d_h,    block + 0, 2);
    std::memcpy(&dmin_h, block + 2, 2);
    float d_f    = fp16_to_fp32(d_h);
    float dmin_f = fp16_to_fp32(dmin_h);

    float sc[8], m[8];
    unpack_scale_min_k4(block + 4, sc, m);

    const uint8_t* qs = block + 16;
    for (int g = 0; g < 4; ++g) {
        const uint8_t* chunk = qs + g * 32;
        uint8x16_t q0 = vld1q_u8(chunk + 0);
        uint8x16_t q1 = vld1q_u8(chunk + 16);

        uint8x16_t lo0 = vandq_u8(q0, vdupq_n_u8(0x0F));
        uint8x16_t hi0 = vshrq_n_u8(q0, 4);
        uint8x16_t lo1 = vandq_u8(q1, vdupq_n_u8(0x0F));
        uint8x16_t hi1 = vshrq_n_u8(q1, 4);

        uint16x8_t lo_w0 = vmovl_u8(vget_low_u8(lo0));
        uint16x8_t lo_w1 = vmovl_u8(vget_high_u8(lo0));
        uint16x8_t lo_w2 = vmovl_u8(vget_low_u8(lo1));
        uint16x8_t lo_w3 = vmovl_u8(vget_high_u8(lo1));

        float32x4_t lo_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_w0)));
        float32x4_t lo_f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_w0)));
        float32x4_t lo_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_w1)));
        float32x4_t lo_f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_w1)));
        float32x4_t lo_f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_w2)));
        float32x4_t lo_f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_w2)));
        float32x4_t lo_f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_w3)));
        float32x4_t lo_f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_w3)));

        uint16x8_t hi_w0 = vmovl_u8(vget_low_u8(hi0));
        uint16x8_t hi_w1 = vmovl_u8(vget_high_u8(hi0));
        uint16x8_t hi_w2 = vmovl_u8(vget_low_u8(hi1));
        uint16x8_t hi_w3 = vmovl_u8(vget_high_u8(hi1));

        float32x4_t hi_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_w0)));
        float32x4_t hi_f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_w0)));
        float32x4_t hi_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_w1)));
        float32x4_t hi_f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_w1)));
        float32x4_t hi_f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_w2)));
        float32x4_t hi_f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_w2)));
        float32x4_t hi_f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_w3)));
        float32x4_t hi_f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_w3)));

        float32x4_t s0 = vdupq_n_f32(d_f * sc[2 * g]);
        float32x4_t s1 = vdupq_n_f32(d_f * sc[2 * g + 1]);
        float32x4_t neg_t0 = vdupq_n_f32(-(dmin_f * m[2 * g]));
        float32x4_t neg_t1 = vdupq_n_f32(-(dmin_f * m[2 * g + 1]));

        int lo_off = g * 64;
        int hi_off = g * 64 + 32;
        vst1q_f32(out + lo_off + 0,  vmlaq_f32(neg_t0, s0, lo_f0));
        vst1q_f32(out + lo_off + 4,  vmlaq_f32(neg_t0, s0, lo_f1));
        vst1q_f32(out + lo_off + 8,  vmlaq_f32(neg_t0, s0, lo_f2));
        vst1q_f32(out + lo_off + 12, vmlaq_f32(neg_t0, s0, lo_f3));
        vst1q_f32(out + lo_off + 16, vmlaq_f32(neg_t0, s0, lo_f4));
        vst1q_f32(out + lo_off + 20, vmlaq_f32(neg_t0, s0, lo_f5));
        vst1q_f32(out + lo_off + 24, vmlaq_f32(neg_t0, s0, lo_f6));
        vst1q_f32(out + lo_off + 28, vmlaq_f32(neg_t0, s0, lo_f7));

        vst1q_f32(out + hi_off + 0,  vmlaq_f32(neg_t1, s1, hi_f0));
        vst1q_f32(out + hi_off + 4,  vmlaq_f32(neg_t1, s1, hi_f1));
        vst1q_f32(out + hi_off + 8,  vmlaq_f32(neg_t1, s1, hi_f2));
        vst1q_f32(out + hi_off + 12, vmlaq_f32(neg_t1, s1, hi_f3));
        vst1q_f32(out + hi_off + 16, vmlaq_f32(neg_t1, s1, hi_f4));
        vst1q_f32(out + hi_off + 20, vmlaq_f32(neg_t1, s1, hi_f5));
        vst1q_f32(out + hi_off + 24, vmlaq_f32(neg_t1, s1, hi_f6));
        vst1q_f32(out + hi_off + 28, vmlaq_f32(neg_t1, s1, hi_f7));
    }
}

#endif  // FRN_NEON

// ---------------------------------------------------------------------------
// Q4_K dequant of a whole tensor (validation path)
// ---------------------------------------------------------------------------

static inline void dequant_q4_k_naive(
    const uint8_t* raw,
    std::size_t n_blocks,
    float* out
) {
    for (std::size_t b = 0; b < n_blocks; ++b) {
        dequant_q4_k_block_naive(raw + b * 144, out + b * 256);
    }
}

// ---------------------------------------------------------------------------
// Q4_K x FP32 matrix-vector multiply
// ---------------------------------------------------------------------------
//
// weight: shape (n, k) packed Q4_K (n rows, k cols of in_features).
// x:      shape (k,) FP32 input activations.
// out:    shape (n,) FP32 output: out[i] = sum_j w[i,j] * x[j]
//
// Multi-threading: split the rows across threads. The caller passes
// the [row_start, row_end) range to avoid needing atomics in the
// output buffer.

#if defined(FRN_NEON)
typedef float (*dot_block_fn)(const uint8_t*, const float*);
#endif

static inline void matmul_q4_k(
    const uint8_t* weight,
    const float* x,
    float* out,
    std::size_t n,
    std::size_t k
) {
    std::size_t n_blocks = k / 256;
#if defined(FRN_NEON)
    dot_block_fn dot_block = dot_q4_k_block_neon;
#else
    auto dot_block = dot_q4_k_block_naive;
#endif
    // Naive single-thread loop kept for correctness / debugging.
    for (std::size_t i = 0; i < n; ++i) {
        const uint8_t* row = weight + i * n_blocks * 144;
        float acc = 0.0f;
        for (std::size_t b = 0; b < n_blocks; ++b) {
            acc += dot_block(row + b * 144, x + b * 256);
        }
        out[i] = acc;
    }
}

#if defined(FRN_NEON)
// Multi-threaded variant. Splits rows across threads; each thread
// computes a contiguous row range and writes its slice of ``out``.
// Memory-bandwidth-bound matmul scales almost linearly with cores
// for the dequant step - the per-row working set fits in L1, so
// contention is limited to the input vector ``x`` which is read-only
// and shared. We use the platform's std::thread pool (one shot,
// spawned per call). For persistent pools we'd want a custom
// scheduler; the call frequency in the forwarder (per-projection
// per-layer) doesn't justify the bookkeeping yet.
#include <thread>
#include <vector>

static inline void matmul_q4_k_mt(
    const uint8_t* weight,
    const float* x,
    float* out,
    std::size_t n,
    std::size_t k,
    int n_threads = 0
) {
    std::size_t n_blocks = k / 256;
    dot_block_fn dot_block = dot_q4_k_block_neon;
    if (n_threads <= 0) {
        n_threads = (int)std::thread::hardware_concurrency();
        if (n_threads < 1) n_threads = 1;
    }
    // Don't spawn more threads than rows - empty ranges are wasted work.
    if ((std::size_t)n_threads > n) n_threads = (int)n;
    if (n_threads <= 1) {
        for (std::size_t i = 0; i < n; ++i) {
            const uint8_t* row = weight + i * n_blocks * 144;
            float acc = 0.0f;
            for (std::size_t b = 0; b < n_blocks; ++b) {
                acc += dot_block(row + b * 144, x + b * 256);
            }
            out[i] = acc;
        }
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    auto worker = [&](std::size_t start, std::size_t end) {
        for (std::size_t i = start; i < end; ++i) {
            const uint8_t* row = weight + i * n_blocks * 144;
            float acc = 0.0f;
            for (std::size_t b = 0; b < n_blocks; ++b) {
                acc += dot_block(row + b * 144, x + b * 256);
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
