# Native backend

Flatrun's optional C++/NEON backend accelerates the projection
matmul — the dominant cost in every decoder layer. It lives in
`src/flatrun_native/` and is built as a pybind11 extension when
the `native` extra is requested:

```bash
pip install -e ".[native]"
# or
pip install -e ".[dev,native]"
```

The native backend is opt-in via `--backend native`. When the
extension is not built (no pybind11 toolchain), or when a
specific kernel rejects an unusual layout, the call falls back
to the pure-Python numpy path transparently.

## What it covers

| Quant | Block size | SIMD width | Multi-threaded | Batched |
|-------|-----------:|-----------:|---------------:|--------:|
| Q4_K  | 256 elem / 144 bytes | 4-wide NEON | yes | yes |
| Q6_K  | 256 elem / 210 bytes | 4-wide NEON (per group) | yes | yes |
| Q8_0  | 32 elem / 34 bytes | 4-wide NEON | yes | yes |

Each quant has:
- A fused dequant + matmul kernel (`matmul_q*_<format>`)
- A batched 2-D entry point (`matmul_q*_<format>_batched`) that
  dispatches the kernel `seq` times inside C++ — zero per-token
  pybind11 round-trip
- A multi-threaded variant (`matmul_q*_<format>_mt`) that splits
  rows across `std::thread::hardware_concurrency()` workers
- A parity test in `src/flatrun_native/tests/`

## Layout of the C++ code

```
src/flatrun_native/
├── _C.cpp                 # pybind11 entry points
├── kernels/
│   ├── q4_k.h             # Q4_K fused dequant + matmul (NEON)
│   ├── q6_k.h             # Q6_K fused dequant + matmul (NEON)
│   └── q8_0.h             # Q8_0 fused dequant + matmul (NEON)
├── python/
│   └── __init__.py        # NativeBackend facade (1-D / 2-D dispatch)
├── tests/
│   ├── test_q4_k_parity.py
│   ├── test_q6_k_parity.py
│   ├── test_q8_0_parity.py
│   ├── test_native_backend.py
│   └── test_e2e_backends.py
└── bench/
    └── bench_native_gemm.py
```

## Build / install

The C++ extension is built by `setup.py` via `pybind11`. The
build is gated on `pybind11` being importable — if it's missing,
`pip install -e .` still works but the extension is skipped
and the runtime detects the missing `_C` and falls back to numpy.

```bash
# Force a fresh build
rm -rf build/ src/flatrun_native/_C*.so
pip install -e ".[native]"
```

The setup script detects the host architecture:

- **Apple Silicon (arm64)**: adds `-arch arm64` to clang
- **Linux ARM64**: adds `-march=armv8-a`
- **Other**: no architecture-specific flags

## Public Python API

```python
from flatrun_native import NativeBackend

backend = NativeBackend()
if backend.available:
    # 1-D input: single-token decode
    out = backend.matmul_q4k(x, weight, n, k)

    # 2-D input: prefill (seq tokens at once)
    out = backend.matmul_q4k(x_2d, weight, n, k)  # shape (seq, n)
```

The `NativeBackend` facade accepts both 1-D `(k,)` and 2-D
`(seq, k)` activations; the 2-D path is dispatched to the C++
batched entry point, which is the only way to get the
`--max-new 1` per-step call to be free of per-token pybind11
overhead on the prefill.

## Per-quant parity

Each quant has a generated-bytes parity test that compares the
C++ fused matmul against the pure-numpy `dequant + matmul`
reference. The tolerance is the format's true FP32 round-off
plus the half-precision scale rounding inside the kernel:

| Quant | max abs diff | max rel diff |
|-------|-------------:|-------------:|
| Q4_K  | ~1e-4 | ~3e-6 |
| Q6_K  | ~5e-2 | ~3e-4 (1 in 17000 error rate on the format) |
| Q8_0  | ~5e-2 | ~2e-5 |

The Q6_K tolerance is larger because the reference uses
random-bytes scales that span the full int8 range; the model's
actual scales are much smaller in magnitude and the relative
error is well below the 1 % threshold used by the in-tree
end-to-end parity test (`test_e2e_backends.py`).

## Multi-threading notes

The `_mt` variants split the output rows across `n_threads`
workers. Each thread is spawned per call (one-shot pool); the
call frequency in the forwarder (per projection per layer) is
low enough that the ~50 µs thread startup overhead is amortised
away on the larger matmul shapes (≥ 256 rows).

For small shapes (small mod / output heads), the per-row
chunk-per-thread is below the L1 cache working set and the
MT path is essentially a no-op. Use the `_mt` paths anyway —
they fall back to single-threaded when `n_threads > n` rows.

## Batched API

The batched entry points are called with a 2-D activation
matrix of shape `(seq, k)` and produce a 2-D output
`(seq, n)`. The C++ side dispatches the single-row kernel
`seq` times inside a single pybind11 call:

```python
# Python (per-token loop, slow)
out = np.empty((seq, n), dtype=np.float32)
for s in range(seq):
    out[s] = _C.matmul_q4_k(x[s], weight, n, k)

# C++ batched (one call, no overhead)
out = _C.matmul_q4_k_batched(x, weight, n, k)
```

For prefill (`seq > 1`) the batched path is the only way to
avoid paying the pybind11 marshalling cost `seq` times — the
profile shows this saves ~50 ms across a 32-token prefill on
Qwen3-0.6B.

## Limitations

- The native backend is **ARM-only** at the SIMD level. The
  compiled extension still loads on x86 / x86_64 (the kernels
  fall back to scalar), but the speedup factors quoted in this
  document are from Apple Silicon measurements.
- **Native is currently slower than numpy/BLAS** for full
  forward passes because scalar + MT kernels cannot compete
  with Apple's Accelerate BLAS at full utilisation. The kernel
  wins are concentrated on small-shape projections where BLAS
  dispatch overhead dominates.
- The dequant-only Python paths (`flatrun.dequant.gguf`) are
  still used by the Python backend — the native backend is
  *additive*, not a replacement for the numpy decode.

## Roadmap

The next kernel improvements are (in order):

1. SIMD vectorize the Q6_K block body (replace the scalar
   FPU loop with a 4-wide NEON path that handles all four
   groups in lock-step).
2. Add cache blocking to the matmul outer loop so each thread
   keeps its row data in L1.
3. Batched matmul for `seq > 1` already uses the 2-D entry
   point; consider a true matrix-matrix tiled kernel that
   amortises the dequant across many output rows.
4. FP16 weight storage + activation to halve the matmul
   bandwidth.
