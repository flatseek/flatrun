// Q6_K GEMM kernel - fused dequant + matrix-vector multiply.
//
// Layout (per 210-byte block, 256 elements):
//   bytes   0..128 : uint8_t ql[128]   (low 4 bits of each value)
//   bytes 128..192 : uint8_t qh[64]    (high 2 bits of each value)
//   bytes 192..208 : int8_t  scales[16] (signed, one per 16 elements)
//   bytes 208..210 : ggml_half d        (super-block scale)
//
// Each 128-element half is divided into 4 strided groups of 32
// elements each. The 16-element stride is what makes the format hard
// to follow without a picture; see ``dequantize_row_q6_K`` in
// llama.cpp for the authoritative reference.
//
// Quant formula:
//   q = (nibble + high_2_bits) - 32   // unsigned 6-bit centred around zero
//   y = d * scales[is + 2*k] * q
//
// SIMD strategy: the four groups read overlapping bits from the same
// ``l`` index, which makes the natural parallelism per group one
// element rather than four. The kernel uses the scalar FPU path
// inside a NEON-enabled translation unit so the compiler can still
// vectorise the surrounding loads; the integer unpacking dominates
// per-element cost anyway. Q4_K and Q8_0 retain their 4-wide NEON
// pipelines because their layouts allow it.

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
// Scalar Q6_K block dot product (used both as the reference and as
// the SIMD body when the format doesn't vectorise cleanly).
// ---------------------------------------------------------------------------

static inline float dot_q6_k_block(
    const uint8_t* block,
    const float* x
) {
    uint16_t d_h;
    std::memcpy(&d_h, block + 208, 2);
    float d = fp16_to_fp32(d_h);

    const uint8_t* ql = block + 0;
    const uint8_t* qh = block + 128;
    const int8_t*  scales = reinterpret_cast<const int8_t*>(block + 192);

    float acc = 0.0f;
    // Two halves of 128 elements each.
    for (int half = 0; half < 2; ++half) {
        const uint8_t* ql_h = ql + half * 64;
        const uint8_t* qh_h = qh + half * 32;
        const int8_t*  sc_h = scales + half * 8;
        int base = half * 128;

        for (int l = 0; l < 32; ++l) {
            int is = (l < 16) ? 0 : 1;
            uint8_t qhb = qh_h[l];

            // group 0: y[base + l]
            {
                int q6 = (ql_h[l] & 0x0F) | (((qhb >> 0) & 3) << 4);
                float q = static_cast<float>(q6) - 32.0f;
                acc += d * static_cast<float>(sc_h[is + 0]) * q * x[base + l];
            }
            // group 1: y[base + l + 32]
            {
                int q6 = (ql_h[l + 32] & 0x0F) | (((qhb >> 2) & 3) << 4);
                float q = static_cast<float>(q6) - 32.0f;
                acc += d * static_cast<float>(sc_h[is + 2]) * q * x[base + l + 32];
            }
            // group 2: y[base + l + 64]
            {
                int q6 = (ql_h[l] >> 4) | (((qhb >> 4) & 3) << 4);
                float q = static_cast<float>(q6) - 32.0f;
                acc += d * static_cast<float>(sc_h[is + 4]) * q * x[base + l + 64];
            }
            // group 3: y[base + l + 96]
            {
                int q6 = (ql_h[l + 32] >> 4) | (((qhb >> 6) & 3) << 4);
                float q = static_cast<float>(q6) - 32.0f;
                acc += d * static_cast<float>(sc_h[is + 6]) * q * x[base + l + 96];
            }
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Naive alias (kept for the dispatch table that picks a block function).
// ---------------------------------------------------------------------------

static inline float dot_q6_k_block_naive(
    const uint8_t* block,
    const float* x
) {
    return dot_q6_k_block(block, x);
}

// ---------------------------------------------------------------------------
// NEON alias - on ARM64 we still call the same scalar body because
// the Q6_K layout doesn't 4-wide vectorise cleanly. Keeping the
// symbol here means the dispatch in matmul_q6_k is uniform across
// targets.
// ---------------------------------------------------------------------------

#if defined(FRN_NEON)
static inline float dot_q6_k_block_neon(
    const uint8_t* block,
    const float* x
) {
    return dot_q6_k_block(block, x);
}
#endif

// ---------------------------------------------------------------------------
// Q6_K dequant of a whole tensor (validation path)
// ---------------------------------------------------------------------------

static inline void dequant_q6_k_naive(
    const uint8_t* raw,
    std::size_t n_blocks,
    float* out
) {
    for (std::size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* block = raw + b * 210;
        uint16_t d_h;
        std::memcpy(&d_h, block + 208, 2);
        float d = fp16_to_fp32(d_h);

        const uint8_t* ql = block + 0;
        const uint8_t* qh = block + 128;
        const int8_t*  scales = reinterpret_cast<const int8_t*>(block + 192);

        for (int half = 0; half < 2; ++half) {
            const uint8_t* ql_h = ql + half * 64;
            const uint8_t* qh_h = qh + half * 32;
            const int8_t*  sc_h = scales + half * 8;
            int base = half * 128;

            for (int l = 0; l < 32; ++l) {
                int is = (l < 16) ? 0 : 1;
                uint8_t qhb = qh_h[l];

                int q0 = (ql_h[l]      & 0x0F) | (((qhb >> 0) & 3) << 4);
                int q1 = (ql_h[l + 32] & 0x0F) | (((qhb >> 2) & 3) << 4);
                int q2 = (ql_h[l]      >> 4)    | (((qhb >> 4) & 3) << 4);
                int q3 = (ql_h[l + 32] >> 4)    | (((qhb >> 6) & 3) << 4);

                out[base + l]      = d * static_cast<float>(sc_h[is + 0]) * (static_cast<float>(q0) - 32.0f);
                out[base + l + 32] = d * static_cast<float>(sc_h[is + 2]) * (static_cast<float>(q1) - 32.0f);
                out[base + l + 64] = d * static_cast<float>(sc_h[is + 4]) * (static_cast<float>(q2) - 32.0f);
                out[base + l + 96] = d * static_cast<float>(sc_h[is + 6]) * (static_cast<float>(q3) - 32.0f);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Q6_K x FP32 matrix-vector multiply
// ---------------------------------------------------------------------------

#if defined(FRN_NEON)
typedef float (*dot_block_q6_k_fn)(const uint8_t*, const float*);
#endif

static inline void matmul_q6_k(
    const uint8_t* weight,
    const float* x,
    float* out,
    std::size_t n,
    std::size_t k
) {
    std::size_t n_blocks = k / 256;
#if defined(FRN_NEON)
    dot_block_q6_k_fn dot_block = dot_q6_k_block_neon;
#else
    auto dot_block = dot_q6_k_block_naive;
#endif
    for (std::size_t i = 0; i < n; ++i) {
        const uint8_t* row = weight + i * n_blocks * 210;
        float acc = 0.0f;
        for (std::size_t b = 0; b < n_blocks; ++b) {
            acc += dot_block(row + b * 210, x + b * 256);
        }
        out[i] = acc;
    }
}

#if defined(FRN_NEON)
#include <thread>
#include <vector>

// Multi-threaded Q6_K matmul: parallel across rows. See matmul_q4_k_mt
// in q4_k.h for the design notes. Q6_K uses the scalar body so the
// thread speedup is the most impactful here.
static inline void matmul_q6_k_mt(
    const uint8_t* weight,
    const float* x,
    float* out,
    std::size_t n,
    std::size_t k,
    int n_threads = 0
) {
    std::size_t n_blocks = k / 256;
#if defined(FRN_NEON)
    dot_block_q6_k_fn dot_block = dot_q6_k_block_neon;
#else
    auto dot_block = dot_q6_k_block_naive;
#endif
    if (n_threads <= 0) {
        n_threads = (int)std::thread::hardware_concurrency();
        if (n_threads < 1) n_threads = 1;
    }
    if ((std::size_t)n_threads > n) n_threads = (int)n;
    if (n_threads <= 1) {
        for (std::size_t i = 0; i < n; ++i) {
            const uint8_t* row = weight + i * n_blocks * 210;
            float acc = 0.0f;
            for (std::size_t b = 0; b < n_blocks; ++b) {
                acc += dot_block(row + b * 210, x + b * 256);
            }
            out[i] = acc;
        }
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    auto worker = [&](std::size_t start, std::size_t end) {
        for (std::size_t i = start; i < end; ++i) {
            const uint8_t* row = weight + i * n_blocks * 210;
            float acc = 0.0f;
            for (std::size_t b = 0; b < n_blocks; ++b) {
                acc += dot_block(row + b * 210, x + b * 256);
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
