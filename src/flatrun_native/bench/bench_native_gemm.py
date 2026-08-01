"""Benchmark: native Q4_K GEMM vs Python dequant + numpy.

Compares three paths on the same Q4_K tensor:

1. Numpy FP32 matmul (the pre-quantisation upper bound) — reference.
2. Python dequant + numpy matmul (the current Flatrun hot path).
3. Native fused Q4_K dequant + matmul (the new ``flatrun_native`` path).

Reports latency, throughput (tokens/sec), and memory footprint.
The reference model is a Qwen2-0.6B-style weight shape:
   - gate/up:  (896, 4864)  -> ~21 700 elements per row
   - down:     (4864, 896)  -> same total
   - q/k/v/o:  (896, 896)  -> 896 elements per row
The benchmark exercises the gate_w shape (the largest decoder weight).

Run:
    PYTHONPATH=src python3 src/flatrun_native/bench/bench_native_gemm.py
"""

from __future__ import annotations

import sys
import os
import time
import struct
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from flatrun.dequant.gguf import dequant_q4_k  # noqa: E402
from flatrun_native import NativeBackend, is_available, is_neon  # noqa: E402


def _make_q4_k_bytes(seed: int, n_blocks: int) -> bytes:
    """Generate valid Q4_K block bytes (parity test helper)."""
    rng = np.random.default_rng(seed)
    out = bytearray()
    for _ in range(n_blocks):
        d = float(rng.uniform(0.01, 1.0))
        dmin = float(rng.uniform(0.0, 0.5))
        out += struct.pack("<ee", d, dmin)
        # 12 bytes of scale/min (kept simple — high 2 bits zero)
        out += bytes(rng.integers(0, 64, size=12, dtype=np.uint8))
        # 128 bytes of packed nibbles (raw, no re-packing)
        out += bytes(rng.integers(0, 16, size=128, dtype=np.uint8))
    return bytes(out)


def bench(name: str, fn, iters: int = 50, warmup: int = 5) -> float:
    """Run ``fn`` ``iters`` times and return the median latency in ms."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    print("=" * 78)
    print("flatrun_native Q4_K GEMM benchmark")
    print("=" * 78)
    print(f"native available: {is_available()}")
    print(f"NEON enabled:     {is_neon()}")
    print()

    if not is_available():
        print("SKIP: native backend not available. Try 'pip install -e .[native]'")
        return

    backend = NativeBackend()

    # Use a representative decoder weight shape. The Qwen2-0.6B
    # gate_w is (4864, 896) on disk; the GGUF tensor pads each row
    # to a multiple of 256 elements, so the actual on-disk layout is
    # (4864, 1024) = 4864 * 1024 = 4 980 224 elements = 19 456 blocks.
    # We benchmark that padded shape directly.
    n = 4864
    k = 1024
    n_blocks = n * k // 256
    raw = _make_q4_k_bytes(seed=42, n_blocks=n_blocks)
    raw_u8 = np.frombuffer(raw, dtype=np.uint8)

    x = np.random.default_rng(0).standard_normal(k).astype(np.float32)

    print(f"weight shape: ({n}, {k})  n_blocks={n_blocks}  raw_size={len(raw)/1024:.1f} KB")
    print(f"activation:   shape=({k},)")
    print()

    # Path 1: numpy FP32 matmul (using a pre-dequantised weight)
    dequantised = dequant_q4_k(raw, (n, k), np.float32)
    print("Path 1: numpy FP32 matmul (pre-dequantised, no dequant cost)")
    t1 = bench("numpy F32 matmul", lambda: dequantised @ x)
    print(f"  median: {t1:.4f} ms")
    print()

    # Path 2: Python dequant + numpy matmul (current Flatrun hot path)
    def py_dequant_then_matmul():
        w = dequant_q4_k(raw, (n, k), np.float32)
        return w @ x
    print("Path 2: Python dequant + numpy matmul (current Flatrun path)")
    t2 = bench("python dequant + matmul", py_dequant_then_matmul)
    print(f"  median: {t2:.4f} ms")
    print()

    # Path 3: Native fused
    print("Path 3: native fused Q4_K dequant + matmul (flatrun_native)")
    t3 = bench("native matmul_q4k", lambda: backend.matmul_q4k(x, raw_u8, n, k))
    print(f"  median: {t3:.4f} ms")
    print()

    # Summary
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  numpy F32 (pre-dequantised):    {t1:8.4f} ms")
    print(f"  Python dequant + numpy:         {t2:8.4f} ms  ({t2 / t1:.2f}x vs F32)")
    print(f"  Native fused (flatrun_native):  {t3:8.4f} ms  ({t3 / t1:.2f}x vs F32, {t3 / t2:.2f}x vs Python)")
    print()
    print(f"  Speedup vs Python: {t2 / t3:.2f}x")
    print(f"  Speedup vs F32:    {t1 / t3:.2f}x")
    print()

    # Memory check
    print("Memory footprint:")
    print(f"  Q4_K raw weight:    {len(raw) / 1024:.1f} KB")
    print(f"  F32 dequantised:    {dequantised.nbytes / 1024:.1f} KB ({dequantised.nbytes / len(raw):.1f}x larger)")
    print(f"  Native (no F32):    {len(raw) / 1024:.1f} KB (zero intermediate)")
    print()

    # Token throughput estimate
    # For a 0.6B model with 24 layers, each layer has 2 large matmuls of
    # similar size (gate_w + up_w + down_w). At 1 token/decode, the
    # total matmul work is ~3 * 24 * t3 per token.
    per_token_ms = 3 * 24 * t3
    print(f"Estimated decode throughput (Qwen2-0.6B Q4_K, single-thread):")
    print(f"  per token: {per_token_ms:.1f} ms")
    print(f"  tokens/sec: {1000.0 / per_token_ms:.2f}")
    print()
    print("Note: this is a single-threaded microbenchmark. The forwarder")
    print("uses 1 thread by default; for multi-threaded throughput gains")
    print("the native kernel needs per-block threading (not yet shipped).")


if __name__ == "__main__":
    main()
