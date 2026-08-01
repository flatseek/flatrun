"""Numerical parity test for the native Q8_0 kernel.

Compares the C++ dequant and fused matmul against the NumPy
reference for randomly generated Q8_0 blocks.

Q8_0 layout (per 34-byte block, 32 elements):
- bytes  0..2  : ggml_half d (FP16 scale)
- bytes  2..34 : int8_t qs[32] (signed 8-bit quant values)
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from flatrun.dequant.gguf import dequant_q8_0  # noqa: E402
from flatrun_native import is_available  # noqa: E402


def _make_q8_0_bytes(seed: int, n_blocks: int) -> bytes:
    """Generate valid Q8_0 block bytes.

    Q8_0 has signed int8 quant values (-128..127) and an FP16
    super-block scale. We randomise within valid ranges.
    """
    rng = np.random.default_rng(seed)
    out = bytearray()
    for _ in range(n_blocks):
        # d: positive FP16 in [0.01, 1.0]
        d = float(rng.uniform(0.01, 1.0))
        out += struct.pack("<e", d)
        # qs: 32 signed int8
        qs = rng.integers(-127, 127, size=32, dtype=np.int8)
        out += bytes(qs)
    return bytes(out)


def test_dequant_q8_0_parity() -> None:
    """Native dequant must match Python dequant to FP32 round-off."""
    if not is_available():
        print("SKIP: native unavailable")
        return
    import flatrun_native._C as _C
    out_dim = 16
    in_dim = 32  # 1 block per row
    n_blocks = out_dim * in_dim // 32  # = 16
    raw = _make_q8_0_bytes(seed=42, n_blocks=n_blocks)
    raw_u8 = np.frombuffer(raw, dtype=np.uint8)

    expected = dequant_q8_0(raw, (out_dim, in_dim), np.float32)
    actual = _C.dequant_q8_0(raw_u8, (out_dim, in_dim))

    diff = np.abs(expected - actual)
    print(f"  shape: {expected.shape}")
    print(f"  max diff: {diff.max():.6e}")
    print(f"  mean diff: {diff.mean():.6e}")
    assert diff.max() < 1e-4, f"diff too large: {diff.max()}"
    print("  OK: parity holds")


def test_matmul_q8_0_parity() -> None:
    """Native Q8_0 fused dequant + matmul must match Python."""
    if not is_available():
        print("SKIP: native unavailable")
        return
    import flatrun_native._C as _C
    # (n, k) = (32, 256). 32*256 = 8192 elements = 256 blocks.
    n = 32
    k = 256
    n_blocks = n * k // 32  # = 256
    raw = _make_q8_0_bytes(seed=7, n_blocks=n_blocks)
    raw_u8 = np.frombuffer(raw, dtype=np.uint8)

    x = np.random.default_rng(99).standard_normal(k).astype(np.float32)

    expected_w = dequant_q8_0(raw, (n, k), np.float32)
    expected = expected_w @ x

    actual = _C.matmul_q8_0(x, raw_u8, n, k)

    diff = np.abs(expected - actual)
    rel = diff / (np.abs(expected) + 1e-6)
    print(f"  n={n}, k={k}")
    print(f"  max diff: {diff.max():.6e}")
    print(f"  mean diff: {diff.mean():.6e}")
    print(f"  max rel:  {rel.max():.6e}")
    assert diff.max() < 5e-2, f"diff too large: {diff.max()}"
    print("  OK: parity holds")


def test_batched_matmul_q8_0_parity() -> None:
    """Batched matmul (2-D input) must match Python per-row."""
    if not is_available():
        print("SKIP: native unavailable")
        return
    import flatrun_native._C as _C
    n = 32
    k = 256
    seq = 4
    n_blocks = n * k // 32
    raw = _make_q8_0_bytes(seed=11, n_blocks=n_blocks)
    raw_u8 = np.frombuffer(raw, dtype=np.uint8)

    rng = np.random.default_rng(123)
    x = rng.standard_normal((seq, k)).astype(np.float32)

    expected_w = dequant_q8_0(raw, (n, k), np.float32)
    expected = x @ expected_w.T  # (seq, n)

    actual = _C.matmul_q8_0_batched(x, raw_u8, n, k)

    diff = np.abs(expected - actual)
    print(f"  seq={seq}, n={n}, k={k}")
    print(f"  max diff: {diff.max():.6e}")
    assert diff.max() < 5e-2, f"diff too large: {diff.max()}"
    print("  OK: batched parity holds")


def main() -> None:
    print("=== test_dequant_q8_0_parity ===")
    test_dequant_q8_0_parity()
    print()
    print("=== test_matmul_q8_0_parity ===")
    test_matmul_q8_0_parity()
    print()
    print("=== test_batched_matmul_q8_0_parity ===")
    test_batched_matmul_q8_0_parity()
    print("\nAll Q8_0 parity tests passed.")


if __name__ == "__main__":
    main()
