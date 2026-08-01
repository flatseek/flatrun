"""Numerical parity test for the native Q6_K kernel.

Compares the C++ dequant and fused matmul against the NumPy
reference for randomly generated Q6_K blocks. The output must
match within FP32 round-off tolerance.

Q6_K layout (per 210-byte block, 256 elements):
- bytes   0..128 : ql[128]   (low 4 bits of each value)
- bytes 128..192 : qh[64]    (high 2 bits of each value)
- bytes 192..208 : scales[16] (signed, one per 16 elements)
- bytes 208..210 : ggml_half d (super-block scale)
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from flatrun.dequant.gguf import dequant_q6_k  # noqa: E402
from flatrun_native import NativeBackend, is_available  # noqa: E402


def _make_q6_k_bytes(seed: int, n_blocks: int) -> bytes:
    """Generate valid Q6_K block bytes.

    Q6_K has 6-bit quant values (0..63) and signed 8-bit scales
    (-128..127). The super-block scale (d) is FP16. We randomise
    all of these within their valid ranges to exercise the
    decoder.
    """
    rng = np.random.default_rng(seed)
    out = bytearray()
    for _ in range(n_blocks):
        # d: positive FP16 in [0.01, 1.0]
        d = float(rng.uniform(0.01, 1.0))
        out += struct.pack("<e", d)
        # ql: 128 bytes (low 4 bits of each value)
        ql = rng.integers(0, 16, size=128, dtype=np.uint8)
        out += bytes(ql)
        # qh: 64 bytes (high 2 bits of each value)
        qh = rng.integers(0, 4, size=64, dtype=np.uint8)
        out += bytes(qh)
        # scales: 16 signed int8
        scales = rng.integers(-127, 127, size=16, dtype=np.int8)
        out += bytes(scales)
    return bytes(out)


def test_dequant_q6_k_parity() -> None:
    """Native dequant must match Python dequant to FP32 round-off."""
    if not is_available():
        print("SKIP: native unavailable")
        return
    backend = NativeBackend()
    # (16, 256) — 16 blocks, 256 elements each.
    out_dim = 16
    in_dim = 256
    n_blocks = out_dim * in_dim // 256  # = 16
    raw = _make_q6_k_bytes(seed=42, n_blocks=n_blocks)
    raw_u8 = np.frombuffer(raw, dtype=np.uint8)

    expected = dequant_q6_k(raw, (out_dim, in_dim), np.float32)
    actual = backend._native if hasattr(backend, "_native") else backend
    # NativeBackend has no dequant_q6k helper — use the C++ extension directly.
    import flatrun_native._C as _C
    actual = _C.dequant_q6_k(raw_u8, (out_dim, in_dim))

    diff = np.abs(expected - actual)
    print(f"  shape: {expected.shape}")
    print(f"  max diff: {diff.max():.6e}")
    print(f"  mean diff: {diff.mean():.6e}")
    assert diff.max() < 1e-3, f"diff too large: {diff.max()}"
    print("  OK: parity holds")


def test_matmul_q6_k_parity() -> None:
    """Native Q6_K fused dequant + matmul must match Python."""
    if not is_available():
        print("SKIP: native unavailable")
        return
    import flatrun_native._C as _C
    n = 32
    k = 512
    n_blocks = n * k // 256  # = 64
    raw = _make_q6_k_bytes(seed=7, n_blocks=n_blocks)
    raw_u8 = np.frombuffer(raw, dtype=np.uint8)

    x = np.random.default_rng(99).standard_normal(k).astype(np.float32)

    expected_w = dequant_q6_k(raw, (n, k), np.float32)
    expected = expected_w @ x

    actual = _C.matmul_q6_k(x, raw_u8, n, k)

    diff = np.abs(expected - actual)
    rel = diff / (np.abs(expected) + 1e-6)
    print(f"  n={n}, k={k}")
    print(f"  max diff: {diff.max():.6e}")
    print(f"  mean diff: {diff.mean():.6e}")
    print(f"  max rel:  {rel.max():.6e}")
    assert diff.max() < 5e-2, f"diff too large: {diff.max()}"
    print("  OK: parity holds")


def test_batched_matmul_q6_k_parity() -> None:
    """Batched matmul (2-D input) must match Python per-row."""
    if not is_available():
        print("SKIP: native unavailable")
        return
    import flatrun_native._C as _C
    n = 32
    k = 512
    seq = 4
    n_blocks = n * k // 256
    raw = _make_q6_k_bytes(seed=11, n_blocks=n_blocks)
    raw_u8 = np.frombuffer(raw, dtype=np.uint8)

    rng = np.random.default_rng(123)
    x = rng.standard_normal((seq, k)).astype(np.float32)

    expected_w = dequant_q6_k(raw, (n, k), np.float32)
    expected = x @ expected_w.T  # (seq, n)

    actual = _C.matmul_q6_k_batched(x, raw_u8, n, k)

    diff = np.abs(expected - actual)
    print(f"  seq={seq}, n={n}, k={k}")
    print(f"  max diff: {diff.max():.6e}")
    assert diff.max() < 5e-2, f"diff too large: {diff.max()}"
    print("  OK: batched parity holds")


def main() -> None:
    print("=== test_dequant_q6_k_parity ===")
    test_dequant_q6_k_parity()
    print()
    print("=== test_matmul_q6_k_parity ===")
    test_matmul_q6_k_parity()
    print()
    print("=== test_batched_matmul_q6_k_parity ===")
    test_batched_matmul_q6_k_parity()
    print("\nAll Q6_K parity tests passed.")


if __name__ == "__main__":
    main()
