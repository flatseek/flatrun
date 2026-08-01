// pybind11 binding for the flatrun_native C++ kernels.
//
// Public Python API (flatrun_native._C):
//
//   dequant_q4_k(weight: np.ndarray[uint8], shape: tuple) -> np.ndarray[float32]
//       Pure dequant of a Q4_K tensor. Returns the F32 buffer matching
//       the on-disk (out, in) layout. Used for parity tests and as a
//       fallback when the GEMM kernel's shape doesn't match the
//       forwarder's expectation.
//
//   matmul_q4_k(x: np.ndarray[float32], weight: np.ndarray[uint8],
//               n: int, k: int) -> np.ndarray[float32]
//       Fused dequant + matmul. ``x`` has shape (k,), ``weight`` is
//       the raw Q4_K payload of a (n, k) row-major tensor. Returns
//       out of shape (n,) where out[i] = sum_j weight[i, j] * x[j].
//       The caller is responsible for choosing the right
//       orientation (the forwarder expresses projections as
//       ``x @ W.T`` which translates to this exact signature).
//
//   is_neon() -> bool
//       True if the extension was built with NEON intrinsics enabled.
//
//   version() -> str
//       Build identifier (git-style hash placeholder).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cstring>

#include "kernels/q4_k.h"
#include "kernels/q6_k.h"
#include "kernels/q8_0.h"

namespace py = pybind11;

namespace flatrun_native {

// ---------------------------------------------------------------------------
// dequant_q4_k
// ---------------------------------------------------------------------------
//
// Python: dequant_q4_k(weight_uint8, shape_tuple) -> float32 ndarray
//   weight_uint8: 1-D ndarray of dtype=np.uint8 (raw block bytes)
//   shape_tuple: (out, in) — must satisfy (out * in) == 256 * n_blocks.

static py::array_t<float> dequant_q4_k_py(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> weight,
    std::tuple<int, int> shape
) {
    int out_dim = std::get<0>(shape);
    int in_dim  = std::get<1>(shape);

    if ((out_dim * in_dim) % 256 != 0) {
        throw std::runtime_error(
            "Q4_K tensor element count must be a multiple of 256"
        );
    }
    std::size_t n_blocks = (out_dim * in_dim) / 256;
    if (static_cast<std::size_t>(weight.size()) != n_blocks * 144) {
        throw std::runtime_error(
            "Q4_K weight byte size mismatch: expected " +
            std::to_string(n_blocks * 144) +
            " bytes, got " +
            std::to_string(weight.size())
        );
    }

    auto out = py::array_t<float>({out_dim, in_dim});
    auto out_mut = out.mutable_unchecked<2>();
    const uint8_t* raw = weight.data();

    // Dequant into a temporary row-major buffer, then we trust the
    // size is small enough that the additional copy is acceptable.
    // The hot path is matmul_q4_k which never materialises the full
    // F32 weight.
    std::vector<float> tmp(n_blocks * 256);
    dequant_q4_k_naive(raw, n_blocks, tmp.data());

    for (int i = 0; i < out_dim; ++i) {
        for (int j = 0; j < in_dim; ++j) {
            out_mut(i, j) = tmp[i * in_dim + j];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// matmul_q4_k
// ---------------------------------------------------------------------------
//
// Python: matmul_q4_k(x, weight_uint8, n, k) -> float32 ndarray
//   x: 1-D ndarray of dtype=np.float32, length k
//   weight_uint8: 1-D ndarray of dtype=np.uint8 (raw Q4_K bytes)
//   n, k: logical dimensions of the (n, k) weight matrix
//
// Returns: 1-D ndarray of shape (n,): out[i] = sum_j weight[i, j] * x[j]

static py::array_t<float> matmul_q4_k_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> weight,
    std::size_t n,
    std::size_t k
) {
    if (k % 256 != 0) {
        throw std::runtime_error(
            "Q4_K matmul: in_features must be a multiple of 256"
        );
    }
    if (static_cast<std::size_t>(x.size()) != k) {
        throw std::runtime_error(
            "Q4_K matmul: x.size() != k"
        );
    }
    std::size_t n_blocks = k / 256;
    if (static_cast<std::size_t>(weight.size()) != n * n_blocks * 144) {
        throw std::runtime_error(
            "Q4_K matmul: weight byte size mismatch"
        );
    }

    auto out = py::array_t<float>({(long)n});
    auto out_mut = out.mutable_unchecked<1>();
    const uint8_t* raw = weight.data();
    const float*   xp  = x.data();

    // Dispatch to the right implementation. The NEON path is in
    // q4_k.h and gets selected on ARM64 at compile time.
    matmul_q4_k(raw, xp, out_mut.mutable_data(0), n, k);

    return out;
}

// ---------------------------------------------------------------------------
// matmul_q4_k_batched
// ---------------------------------------------------------------------------
//
// Python: matmul_q4_k_batched(x, weight, n, k) -> float32 ndarray
//   x: 2-D ndarray of dtype=np.float32, shape (seq, k)
//   weight: 1-D ndarray of dtype=np.uint8 (raw Q4_K bytes)
//   n, k: logical dimensions of the (n, k) weight matrix
//
// Returns: 2-D ndarray of shape (seq, n):
//   out[s, i] = sum_j weight[i, j] * x[s, j]
//
// The forwarder previously called ``matmul_q4_k`` once per token in a
// Python loop; for prefill (seq=10-32) the round-trip cost of the
// pybind11 boundary added more than the kernel itself. The batched
// entry point accepts a 2-D activation matrix and dispatches the
// kernel ``seq`` times inside C++ - same total work, zero per-call
// overhead. Used by the per-quant dispatch in
// ``flatrun.runtime.backend.NativeBackend``.

static py::array_t<float> matmul_q4_k_batched_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> weight,
    std::size_t n,
    std::size_t k
) {
    if (k % 256 != 0) {
        throw std::runtime_error(
            "Q4_K matmul (batched): in_features must be a multiple of 256"
        );
    }
    py::buffer_info x_buf = x.request();
    if (x_buf.ndim != 2) {
        throw std::runtime_error(
            "Q4_K matmul (batched): x must be 2-D (seq, k)"
        );
    }
    std::size_t seq = x_buf.shape[0];
    if (static_cast<std::size_t>(x_buf.shape[1]) != k) {
        throw std::runtime_error(
            "Q4_K matmul (batched): x.shape[1] != k"
        );
    }
    std::size_t n_blocks = k / 256;
    if (static_cast<std::size_t>(weight.size()) != n * n_blocks * 144) {
        throw std::runtime_error(
            "Q4_K matmul (batched): weight byte size mismatch"
        );
    }

    auto out = py::array_t<float>({(long)seq, (long)n});
    auto out_mut = out.mutable_unchecked<2>();
    const uint8_t* raw = weight.data();
    const float*   xp  = static_cast<const float*>(x_buf.ptr);

    for (std::size_t s = 0; s < seq; ++s) {
#if defined(FRN_NEON)
        matmul_q4_k_mt(raw, xp + s * k, &out_mut(s, 0), n, k);
#else
        matmul_q4_k(raw, xp + s * k, &out_mut(s, 0), n, k);
#endif
    }
    return out;
}

static py::array_t<float> matmul_q8_0_batched_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> weight,
    std::size_t n,
    std::size_t k
) {
    if (k % 32 != 0) {
        throw std::runtime_error(
            "Q8_0 matmul (batched): in_features must be a multiple of 32"
        );
    }
    py::buffer_info x_buf = x.request();
    if (x_buf.ndim != 2) {
        throw std::runtime_error(
            "Q8_0 matmul (batched): x must be 2-D (seq, k)"
        );
    }
    std::size_t seq = x_buf.shape[0];
    if (static_cast<std::size_t>(x_buf.shape[1]) != k) {
        throw std::runtime_error(
            "Q8_0 matmul (batched): x.shape[1] != k"
        );
    }
    std::size_t n_blocks = k / 32;
    if (static_cast<std::size_t>(weight.size()) != n * n_blocks * 34) {
        throw std::runtime_error(
            "Q8_0 matmul (batched): weight byte size mismatch"
        );
    }

    auto out = py::array_t<float>({(long)seq, (long)n});
    auto out_mut = out.mutable_unchecked<2>();
    const uint8_t* raw = weight.data();
    const float*   xp  = static_cast<const float*>(x_buf.ptr);

    for (std::size_t s = 0; s < seq; ++s) {
#if defined(FRN_NEON)
        matmul_q8_0_mt(raw, xp + s * k, &out_mut(s, 0), n, k);
#else
        matmul_q8_0(raw, xp + s * k, &out_mut(s, 0), n, k);
#endif
    }
    return out;
}

static py::array_t<float> matmul_q6_k_batched_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> weight,
    std::size_t n,
    std::size_t k
) {
    if (k % 256 != 0) {
        throw std::runtime_error(
            "Q6_K matmul (batched): in_features must be a multiple of 256"
        );
    }
    py::buffer_info x_buf = x.request();
    if (x_buf.ndim != 2) {
        throw std::runtime_error(
            "Q6_K matmul (batched): x must be 2-D (seq, k)"
        );
    }
    std::size_t seq = x_buf.shape[0];
    if (static_cast<std::size_t>(x_buf.shape[1]) != k) {
        throw std::runtime_error(
            "Q6_K matmul (batched): x.shape[1] != k"
        );
    }
    std::size_t n_blocks = k / 256;
    if (static_cast<std::size_t>(weight.size()) != n * n_blocks * 210) {
        throw std::runtime_error(
            "Q6_K matmul (batched): weight byte size mismatch"
        );
    }

    auto out = py::array_t<float>({(long)seq, (long)n});
    auto out_mut = out.mutable_unchecked<2>();
    const uint8_t* raw = weight.data();
    const float*   xp  = static_cast<const float*>(x_buf.ptr);

    for (std::size_t s = 0; s < seq; ++s) {
        matmul_q6_k(raw, xp + s * k, &out_mut(s, 0), n, k);
    }
    return out;
}

static py::array_t<float> dequant_q8_0_py(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> weight,
    std::tuple<int, int> shape
) {
    int out_dim = std::get<0>(shape);
    int in_dim  = std::get<1>(shape);

    if ((out_dim * in_dim) % 32 != 0) {
        throw std::runtime_error(
            "Q8_0 tensor element count must be a multiple of 32"
        );
    }
    std::size_t n_blocks = (out_dim * in_dim) / 32;
    if (static_cast<std::size_t>(weight.size()) != n_blocks * 34) {
        throw std::runtime_error(
            "Q8_0 weight byte size mismatch: expected " +
            std::to_string(n_blocks * 34) +
            " bytes, got " +
            std::to_string(weight.size())
        );
    }

    auto out = py::array_t<float>({out_dim, in_dim});
    auto out_mut = out.mutable_unchecked<2>();
    const uint8_t* raw = weight.data();

    std::vector<float> tmp(n_blocks * 32);
    dequant_q8_0_naive(raw, n_blocks, tmp.data());

    for (int i = 0; i < out_dim; ++i) {
        for (int j = 0; j < in_dim; ++j) {
            out_mut(i, j) = tmp[i * in_dim + j];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// matmul_q8_0
// ---------------------------------------------------------------------------

static py::array_t<float> matmul_q8_0_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> weight,
    std::size_t n,
    std::size_t k
) {
    if (k % 32 != 0) {
        throw std::runtime_error(
            "Q8_0 matmul: in_features must be a multiple of 32"
        );
    }
    if (static_cast<std::size_t>(x.size()) != k) {
        throw std::runtime_error(
            "Q8_0 matmul: x.size() != k"
        );
    }
    std::size_t n_blocks = k / 32;
    if (static_cast<std::size_t>(weight.size()) != n * n_blocks * 34) {
        throw std::runtime_error(
            "Q8_0 matmul: weight byte size mismatch"
        );
    }

    auto out = py::array_t<float>({(long)n});
    auto out_mut = out.mutable_unchecked<1>();
    const uint8_t* raw = weight.data();
    const float*   xp  = x.data();

    matmul_q8_0(raw, xp, out_mut.mutable_data(0), n, k);

    return out;
}

// ---------------------------------------------------------------------------
// dequant_q6_k
// ---------------------------------------------------------------------------

static py::array_t<float> dequant_q6_k_py(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> weight,
    std::tuple<int, int> shape
) {
    int out_dim = std::get<0>(shape);
    int in_dim  = std::get<1>(shape);

    if ((out_dim * in_dim) % 256 != 0) {
        throw std::runtime_error(
            "Q6_K tensor element count must be a multiple of 256"
        );
    }
    std::size_t n_blocks = (out_dim * in_dim) / 256;
    if (static_cast<std::size_t>(weight.size()) != n_blocks * 210) {
        throw std::runtime_error(
            "Q6_K weight byte size mismatch: expected " +
            std::to_string(n_blocks * 210) +
            " bytes, got " +
            std::to_string(weight.size())
        );
    }

    auto out = py::array_t<float>({out_dim, in_dim});
    auto out_mut = out.mutable_unchecked<2>();
    const uint8_t* raw = weight.data();

    std::vector<float> tmp(n_blocks * 256);
    dequant_q6_k_naive(raw, n_blocks, tmp.data());

    for (int i = 0; i < out_dim; ++i) {
        for (int j = 0; j < in_dim; ++j) {
            out_mut(i, j) = tmp[i * in_dim + j];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// matmul_q6_k
// ---------------------------------------------------------------------------

static py::array_t<float> matmul_q6_k_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> weight,
    std::size_t n,
    std::size_t k
) {
    if (k % 256 != 0) {
        throw std::runtime_error(
            "Q6_K matmul: in_features must be a multiple of 256"
        );
    }
    if (static_cast<std::size_t>(x.size()) != k) {
        throw std::runtime_error(
            "Q6_K matmul: x.size() != k"
        );
    }
    std::size_t n_blocks = k / 256;
    if (static_cast<std::size_t>(weight.size()) != n * n_blocks * 210) {
        throw std::runtime_error(
            "Q6_K matmul: weight byte size mismatch"
        );
    }

    auto out = py::array_t<float>({(long)n});
    auto out_mut = out.mutable_unchecked<1>();
    const uint8_t* raw = weight.data();
    const float*   xp  = x.data();

    matmul_q6_k(raw, xp, out_mut.mutable_data(0), n, k);

    return out;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

static bool is_neon() {
#if defined(FRN_NEON)
    return true;
#else
    return false;
#endif
}

static std::string version() {
    return "flatrun_native 0.1.0 (Q4_K, Q6_K, Q8_0 naive + NEON)";
}

}  // namespace flatrun_native

PYBIND11_MODULE(_C, m) {
    m.doc() = "flatrun_native C++ kernels — fused Q4_K dequant + matmul";
    m.def("dequant_q4_k", &flatrun_native::dequant_q4_k_py,
          "Dequant a Q4_K tensor to FP32",
          py::arg("weight"), py::arg("shape"));
    m.def("matmul_q4_k", &flatrun_native::matmul_q4_k_py,
          "Fused Q4_K dequant + matrix-vector multiply",
          py::arg("x"), py::arg("weight"), py::arg("n"), py::arg("k"));
    m.def("matmul_q4_k_batched", &flatrun_native::matmul_q4_k_batched_py,
          "Batched fused Q4_K dequant + matrix-matrix multiply",
          py::arg("x"), py::arg("weight"), py::arg("n"), py::arg("k"));
    m.def("dequant_q8_0", &flatrun_native::dequant_q8_0_py,
          "Dequant a Q8_0 tensor to FP32",
          py::arg("weight"), py::arg("shape"));
    m.def("matmul_q8_0", &flatrun_native::matmul_q8_0_py,
          "Fused Q8_0 dequant + matrix-vector multiply",
          py::arg("x"), py::arg("weight"), py::arg("n"), py::arg("k"));
    m.def("matmul_q8_0_batched", &flatrun_native::matmul_q8_0_batched_py,
          "Batched fused Q8_0 dequant + matrix-matrix multiply",
          py::arg("x"), py::arg("weight"), py::arg("n"), py::arg("k"));
    m.def("dequant_q6_k", &flatrun_native::dequant_q6_k_py,
          "Dequant a Q6_K tensor to FP32",
          py::arg("weight"), py::arg("shape"));
    m.def("matmul_q6_k", &flatrun_native::matmul_q6_k_py,
          "Fused Q6_K dequant + matrix-vector multiply",
          py::arg("x"), py::arg("weight"), py::arg("n"), py::arg("k"));
    m.def("matmul_q6_k_batched", &flatrun_native::matmul_q6_k_batched_py,
          "Batched fused Q6_K dequant + matrix-matrix multiply",
          py::arg("x"), py::arg("weight"), py::arg("n"), py::arg("k"));
    m.def("is_neon", &flatrun_native::is_neon,
          "True if the extension was built with ARM NEON intrinsics");
    m.def("version", &flatrun_native::version,
          "Build identifier");
}
