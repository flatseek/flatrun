"""Numerical parity test for the native Q4_K kernel.

Compares the C++ dequant against the Python reference for randomly
generated Q4_K blocks. The output must match within FP32 round-off
tolerance (~1e-4 absolute). Also covers the matmul path with a
random input against the equivalent ``dequant + x @ W.T`` path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from flatrun.dequant.gguf import dequant_q4_k  # noqa: E402
from flatrun_native import NativeBackend, is_available  # noqa: E402


def _make_q4_k_bytes(seed: int, n_blocks: int) -> bytes:
    """Generate valid Q4_K block bytes.

    Random bytes would produce NaN FP16 scales and confuse the
    decoder. We hand-craft each block so the FP16 scale/dmin land
    in the positive normalised range, the 6-bit scale/min values
    are valid, and the packed nibbles are uniformly 0..15.
    """
    import struct
    rng = np.random.default_rng(seed)
    out = bytearray()
    for _ in range(n_blocks):
        # d, dmin: positive FP16 in [0.01, 1.0]
        d = float(rng.uniform(0.01, 1.0))
        dmin = float(rng.uniform(0.0, 0.5))
        out += struct.pack("<ee", d, dmin)  # 'e' is half-float pack
        # 8 pairs of 6-bit scale/min, packed in 12 bytes
        # j < 4: d = q[j] & 63, m = q[j+4] & 63
        # j >= 4: bits split across bytes
        scales = rng.integers(0, 64, size=8, dtype=np.uint8)
        mins = rng.integers(0, 64, size=8, dtype=np.uint8)
        # Encode the 12 bytes per the upstream GGUF spec.
        q = np.zeros(12, dtype=np.uint8)
        for j in range(4):
            q[j] = scales[j]
            q[j + 4] = mins[j]
        for j in range(4):
            q[j + 8] = ((mins[j + 4] & 0x0F) << 4) | (scales[j + 4] & 0x0F)
            # The high 2 bits of (scales[j+4], mins[j+4]) are stored in
            # q[j] and q[j+4] respectively. We don't worry about the
            # exact mapping for the parity test — we just need the
            # decoder to produce the same output for both Python and
            # C++ paths.
        # Actually, the cleanest is to just use the Python decoder's
        # logic: high 2 bits of (sc[4..7], m[4..7]) live in (q[0..3], q[4..7]).
        # For the test, we keep them zero so the unpack is unambiguous.
        out += bytes(q)
        # 128 bytes of packed nibbles 0..15. Each byte holds a low
        # nibble (elements 0..15 of the 32-byte chunk) and a high
        # nibble (elements 16..31). The 32-byte chunk is split into
        # 4 such chunks of 32 bytes each (128 bytes total).
        qs = rng.integers(0, 16, size=128, dtype=np.uint8)
        # The qs array already has 128 bytes; we keep it as-is.
        out += bytes(qs.astype(np.uint8))
    return bytes(out)


def test_dequant_q4_k_parity() -> None:
    """Native dequant must match Python dequant to FP32 round-off."""
    if not is_available():
        print("SKIP: native unavailable")
        return
    backend = NativeBackend()
    # Shape: (16, 256) — 16 output features, 256 in_features.
    # 16 * 256 = 4096 elements = 16 blocks.
    out_dim = 16
    in_dim = 256
    n_blocks = out_dim * in_dim // 256  # = 16
    raw = _make_q4_k_bytes(seed=42, n_blocks=n_blocks)
    raw_u8 = np.frombuffer(raw, dtype=np.uint8)

    # Python reference
    expected = dequant_q4_k(raw, (out_dim, in_dim), np.float32)

    # Native
    actual = backend.dequant_q4k(raw_u8, (out_dim, in_dim))

    diff = np.abs(expected - actual)
    print(f"  shape: {expected.shape}")
    print(f"  max diff: {diff.max():.6e}")
    print(f"  mean diff: {diff.mean():.6e}")
    assert diff.max() < 1e-4, f"diff too large: {diff.max()}"
    print("  OK: parity holds")


def test_matmul_q4_k_parity() -> None:
    """Native matmul must match Python dequant+matmul to FP32 round-off."""
    if not is_available():
        print("SKIP: native unavailable")
        return
    backend = NativeBackend()
    # (n, k) = (32, 512). 32*512 = 16384 elements = 64 blocks.
    n = 32
    k = 512
    n_blocks = n * k // 256  # = 64
    raw = _make_q4_k_bytes(seed=7, n_blocks=n_blocks)
    raw_u8 = np.frombuffer(raw, dtype=np.uint8)

    x = np.random.default_rng(99).standard_normal(k).astype(np.float32)

    # Python: dequant + matmul
    expected_w = dequant_q4_k(raw, (n, k), np.float32)
    expected = expected_w @ x

    # Native
    actual = backend.matmul_q4k(x, raw_u8, n, k)

    diff = np.abs(expected - actual)
    rel = diff / (np.abs(expected) + 1e-6)
    print(f"  n={n}, k={k}")
    print(f"  max diff: {diff.max():.6e}")
    print(f"  mean diff: {diff.mean():.6e}")
    print(f"  max rel:  {rel.max():.6e}")
    assert diff.max() < 1e-2, f"diff too large: {diff.max()}"
    print("  OK: parity holds")


def main() -> None:
    print("=== test_dequant_q4_k_parity ===")
    test_dequant_q4_k_parity()
    print()
    print("=== test_matmul_q4_k_parity ===")
    test_matmul_q4_k_parity()
    print("\nAll parity tests passed.")


if __name__ == "__main__":
    main()
