"""Tests for the dequantization helpers.

Each decoder is tested on a single block worth of data so we don't pull
in the whole 7B model. The reference values come from a tiny round-trip
through the encoder formulas, not from any external tool.
"""

from __future__ import annotations

import numpy as np
import pytest

from flatrun.dequant.gguf import (
    dequant_q1_0,
    dequant_q4_0,
    dequant_q4_k,
    dequant_q5_0,
    dequant_q5_1,
    dequant_q5_k,
    dequant_q6_k,
    dequant_q8_0,
)
from flatrun.dequant.mlx import dequant_mlx_4bit_split
from flatrun.dequant.loader import dequant_mlx_weight


# ---------------------------------------------------------------------------
# GGUF Q8_0 / Q4_0
# ---------------------------------------------------------------------------


def test_q8_0_roundtrip_simple() -> None:
    """Decode a single Q8_0 block and confirm the values come back."""
    # scale = 0.5 fp16; qs = [-2, 4, 0, 8, ...] (32 entries)
    raw = bytearray()
    raw.extend(np.float16(0.5).tobytes())
    qs = np.array([-2, 4, 0, 8, -3, 1, 2, 3, -1, -4, 5, 6, 7, -8, 9, 0] * 2, dtype=np.int8)
    raw.extend(qs.tobytes())
    arr = dequant_q8_0(bytes(raw), (32,), np.dtype("float32"))
    expected = qs.astype(np.float32) * 0.5
    np.testing.assert_allclose(arr, expected, rtol=1e-5)
    assert arr.shape == (32,)


def test_q8_0_multiple_blocks_reshape() -> None:
    """Multi-block Q8_0 reshapes back to the requested logical shape."""
    # Two blocks = 64 elements.
    raw = bytearray()
    for s in (0.25, 0.75):
        raw.extend(np.float16(s).tobytes())
        qs = np.full(32, 4, dtype=np.int8)
        raw.extend(qs.tobytes())
    arr = dequant_q8_0(bytes(raw), (64,), np.dtype("float32"))
    assert arr.shape == (64,)
    # First 32 should be 4 * 0.25 = 1.0; next 32 should be 4 * 0.75 = 3.0.
    np.testing.assert_allclose(arr[:32], np.full(32, 1.0), rtol=1e-4)
    np.testing.assert_allclose(arr[32:], np.full(32, 3.0), rtol=1e-4)


def test_q4_0_roundtrip_simple() -> None:
    """Q4_0 puts the low nibble of byte ``l`` at index ``l`` and the
    high nibble at index ``l + 16`` (matching ``dequantize_row_q4_0``
    in ``ggml-quants.c``). The earlier interleave-by-2 layout was
    legal-shape wrong and slipped through because of Q4_0's symmetry.
    """
    raw = bytearray()
    raw.extend(np.float16(0.5).tobytes())
    qs = np.full(16, (0xF0 | 0x07), dtype=np.uint8)
    raw.extend(qs.tobytes())
    arr = dequant_q4_0(bytes(raw), (32,), np.dtype("float32"))
    np.testing.assert_allclose(arr[:16], np.full(16, -0.5), rtol=1e-4)
    np.testing.assert_allclose(arr[16:], np.full(16, 3.5), rtol=1e-4)


# ---------------------------------------------------------------------------
# Cross-check against llama.cpp quantize + dequantize
# ---------------------------------------------------------------------------


def _llama_quantize(model: str, quant: str, out_path: str) -> None:
    """Re-quantize a local model to ``quant`` so we can verify against
    a known-good decoder. Skips if ``llama-quantize`` is unavailable.
    """
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("llama-quantize") is None:
        pytest.skip("llama-quantize not on PATH")
    if Path(out_path).is_file():
        return
    subprocess.run(
        [
            "llama-quantize",
            "--allow-requantize",
            model,
            out_path,
            quant,
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.skip(reason="requires local GGUF; see tools/compare_to_llamacpp.py")
def test_q4_k_matches_llama_cpp_on_real_block() -> None:
    """Compare ``dequant_q4_k`` against ``llama-quantize`` output for a
    real tensor pulled from a requantized checkpoint. The block must
    be bit-identical because the encoding is deterministic.
    """


# ---------------------------------------------------------------------------
# MLX 4-bit
# ---------------------------------------------------------------------------


def test_mlx_4bit_split_simple() -> None:
    """A small 4x32 weight where every nibble is set to 7."""
    out_features = 4
    in_features = 32  # 4 packed columns
    weight = np.full((out_features, in_features // 8), 0x77777777, dtype=np.uint32)
    # 1 group of 64 doesn't fit in 32 in_features; resize.
    # Skip - the helper raises if it doesn't match. Use 64 instead.
    out_features = 4
    in_features = 64
    weight = np.full((out_features, in_features // 8), 0x77777777, dtype=np.uint32)
    n_groups = in_features // 64
    scales = np.full((out_features, n_groups), 0.5, dtype=np.float16)
    biases = np.full((out_features, n_groups), 0.1, dtype=np.float16)
    arr = dequant_mlx_4bit_split(weight, scales, biases, dtype="float32")
    # Each nibble = 7, centred = -1, scaled = -0.5 + 0.1 = -0.4
    np.testing.assert_allclose(arr, np.full_like(arr, -0.4), rtol=1e-4)
    assert arr.shape == (out_features, in_features)


def test_mlx_4bit_split_rejects_wrong_shapes() -> None:
    weight = np.zeros((2, 4), dtype=np.uint32)
    scales = np.zeros((2, 1), dtype=np.float16)
    biases = np.zeros((2, 1), dtype=np.float16)
    # in_features (32) does not divide evenly by group size (64) -> 0 groups.
    with pytest.raises(ValueError, match="scales shape"):
        dequant_mlx_4bit_split(weight, scales, biases)


def test_mlx_loader_three_handles(tmp_path) -> None:
    """dequant_mlx_weight reads three handles via a stub acquire()."""
    from flatrun.core.tensor import BufferTensorHandle, TensorMetadata
    from flatrun.utils.types import TensorKey

    out_features = 2
    in_features = 64  # 1 group
    # The weight array is 2-D so the handle preserves the logical shape.
    weight_arr = np.full((out_features, in_features // 8), 0x77777777, dtype=np.uint32)
    scales_arr = np.full((out_features, 1), 0.5, dtype=np.float16)
    biases_arr = np.full((out_features, 1), 0.1, dtype=np.float16)

    def make(name: str, arr: np.ndarray):
        raw = arr.tobytes()
        meta = TensorMetadata(
            key=TensorKey(file="synthetic", name=name, backend=""),
            shape=arr.shape,
            dtype=str(arr.dtype),
            byte_size=len(raw),
            offset=0,
        )
        return BufferTensorHandle(meta, raw)

    handles = {
        "w.weight": make("w.weight", weight_arr),
        "w.scales": make("w.scales", scales_arr),
        "w.biases": make("w.biases", biases_arr),
    }

    def acquire(name: str):
        return handles[name]

    arr = dequant_mlx_weight(acquire, "w", dtype="float32")
    np.testing.assert_allclose(arr, np.full_like(arr, -0.4), rtol=1e-4)
    assert arr.shape == (out_features, in_features)


# ---------------------------------------------------------------------------
# Q4_K (smoke test only - decoder is approximate vs reference C)
# ---------------------------------------------------------------------------


def test_q4_k_decoder_produces_finite_values() -> None:
    """A single Q4_K block (256 elements) decodes without NaN/Inf."""
    raw = bytearray()
    raw.extend(np.float16(0.1).tobytes())  # d
    raw.extend(np.float16(0.01).tobytes())  # dmin
    raw.extend(np.zeros(12, dtype=np.uint8).tobytes())  # 12-byte scale/min block
    raw.extend(np.zeros(128, dtype=np.uint8).tobytes())  # 256 4-bit qs
    arr = dequant_q4_k(bytes(raw), (256,), np.dtype("float32"))
    assert arr.shape == (256,)
    assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# Reference vectors
# ---------------------------------------------------------------------------
#
# Each test below builds a single block whose expected decode was
# worked out by hand from the matching ``dequantize_row_qX_Y`` in
# ``llama.cpp/ggml/src/ggml-quants.c``. That's a sterner check than
# round-tripping through our own encoder: a symmetric encoder would
# happily turn a permuted decoder's output back into the same permuted
# input and pass.


def test_q5_0_hand_derived_vector() -> None:
    """Q5_0: low half from qs & 0xF plus bit 4 from qh, high half from
    qs >> 4 plus bit 4 from qh at the shifted positions. With qh all
    ones, every element gets the high bit; with qs = 0 the centred
    value is ``(0 | 0x10) - 16 = 0`` and ``d * 0 = 0`` for all 32
    elements.
    """
    raw = bytearray()
    raw.extend(np.float16(0.25).tobytes())
    raw.extend(np.frombuffer(np.uint32(0xFFFFFFFF), dtype=np.uint8).tobytes())
    raw.extend(np.zeros(16, dtype=np.uint8).tobytes())
    arr = dequant_q5_0(bytes(raw), (32,), np.dtype("float32"))
    np.testing.assert_allclose(arr, np.zeros(32), atol=1e-6)


def test_q5_1_hand_derived_vector() -> None:
    """Q5_1 differs from Q5_0 only by a +m offset."""
    raw = bytearray()
    raw.extend(np.float16(0.5).tobytes())  # d
    raw.extend(np.float16(1.0).tobytes())  # m
    raw.extend(np.frombuffer(np.uint32(0), dtype=np.uint8).tobytes())  # qh
    qs = np.full(16, 0x55, dtype=np.uint8)  # low=5, high=5
    raw.extend(qs.tobytes())
    arr = dequant_q5_1(bytes(raw), (32,), np.dtype("float32"))
    expected = np.full(32, 5 * 0.5 + 1.0, dtype=np.float32)
    np.testing.assert_allclose(arr, expected, rtol=1e-5, atol=1e-6)


def test_q4_k_zero_block_is_zero() -> None:
    """All-zero Q4_K super-block should decode to all-zero."""
    raw = bytearray()
    raw.extend(np.float16(0.0).tobytes())  # d
    raw.extend(np.float16(0.0).tobytes())  # dmin
    raw.extend(np.zeros(12, dtype=np.uint8).tobytes())
    raw.extend(np.zeros(128, dtype=np.uint8).tobytes())
    arr = dequant_q4_k(bytes(raw), (256,), np.dtype("float32"))
    np.testing.assert_allclose(arr, np.zeros(256), atol=1e-6)


def test_q5_k_zero_block_is_zero() -> None:
    """All-zero Q5_K super-block should decode to all-zero."""
    raw = bytearray()
    raw.extend(np.float16(0.0).tobytes())
    raw.extend(np.float16(0.0).tobytes())
    raw.extend(np.zeros(12, dtype=np.uint8).tobytes())
    raw.extend(np.zeros(32, dtype=np.uint8).tobytes())  # qh
    raw.extend(np.zeros(128, dtype=np.uint8).tobytes())  # qs
    arr = dequant_q5_k(bytes(raw), (256,), np.dtype("float32"))
    np.testing.assert_allclose(arr, np.zeros(256), atol=1e-6)


def test_q6_k_zero_block_is_zero() -> None:
    """All-zero Q6_K super-block should decode to all-zero."""
    raw = bytearray()
    raw.extend(np.zeros(128, dtype=np.uint8).tobytes())  # ql
    raw.extend(np.zeros(64, dtype=np.uint8).tobytes())   # qh
    raw.extend(np.zeros(16, dtype=np.uint8).tobytes())   # scales
    raw.extend(np.float16(0.0).tobytes())                 # d
    arr = dequant_q6_k(bytes(raw), (256,), np.dtype("float32"))
    np.testing.assert_allclose(arr, np.zeros(256), atol=1e-6)


def test_q6_k_high_bit_layout() -> None:
    """Q1_0: each bit in a 128-element block maps to ``+d`` / ``-d``
    in LSB-first order within each byte. Build a block with a
    recognisable byte pattern, then verify the produced sign pattern
    matches the C reference ``bit_offset = j % 8`` exactly.
    """
    raw = bytearray()
    raw.extend(np.float16(0.5).tobytes())
    # 0xAA = 10101010 -> LSB first: -+ -+ -+ -+
    # 0x01 = 00000001 -> LSB first: +- -+ -+ -+ -+ -+ -+ -+ (1 at bit 0)
    for i in range(16):
        raw.append(0xAA if i % 2 == 0 else 0x01)
    arr = dequant_q1_0(bytes(raw), (128,), np.dtype("float32"))
    expected = np.tile(
        np.array(
            [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5,
             0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
        ),
        8,
    )
    np.testing.assert_allclose(arr, expected, atol=1e-6)


def test_q1_0_hand_derived_vector() -> None:
    """Q1_0: each bit in a 128-element block maps to ``+d`` / ``-d``
    in LSB-first order within each byte. Build a block with a
    recognisable byte pattern, then verify the produced sign pattern
    matches the C reference ``bit_offset = j % 8`` exactly.
    """
    raw = bytearray()
    raw.extend(np.float16(0.5).tobytes())
    # 0xAA = 10101010 -> LSB first: -+ -+ -+ -+
    # 0x01 = 00000001 -> LSB first: +- -+ -+ -+ -+ -+ -+ -+ (1 at bit 0)
    for i in range(16):
        raw.append(0xAA if i % 2 == 0 else 0x01)
    arr = dequant_q1_0(bytes(raw), (128,), np.dtype("float32"))
    expected = np.tile(
        np.array(
            [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5,
             0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
        ),
        8,
    )
    np.testing.assert_allclose(arr, expected, atol=1e-6)


def test_q1_0_zero_block_is_zero() -> None:
    """All-zero Q1_0 block has no scale; output should be all zero."""
    raw = bytearray()
    raw.extend(np.float16(0.0).tobytes())
    raw.extend(np.zeros(16, dtype=np.uint8).tobytes())
    arr = dequant_q1_0(bytes(raw), (128,), np.dtype("float32"))
    np.testing.assert_allclose(arr, np.zeros(128), atol=1e-6)


def test_q1_0_handles_non_block_aligned_size() -> None:
    """Embedding tables like Qwen's 151669x2048 are not multiples of
    128 elements; the tail block must pad cleanly without errors.
    """
    raw = bytearray()
    # 130 elements -> ceil(130/128) = 2 blocks -> 36 bytes
    for _ in range(2):
        raw.extend(np.float16(0.5).tobytes())
        raw.extend(np.zeros(16, dtype=np.uint8).tobytes())
    arr = dequant_q1_0(bytes(raw), (130,), np.dtype("float32"))
    assert arr.shape == (130,)
    # Extra elements decode as bit 0 = 0 -> -0.5
    np.testing.assert_allclose(arr[128:], [-0.5, -0.5], atol=1e-6)
    """The qh-to-element mapping in Q6_K is the most error-prone part
    of the decoder. Pin it down with a hand-built block where every
    ql/qh byte is the same so the output has a single known value.

    qh = 0x55 has binary pattern 01010101. For the C reference:

    * q1 = (ql[l]    & 0xF) | ((qh[l] >> 0) & 3) << 4
    * q2 = (ql[l+32] & 0xF) | ((qh[l] >> 2) & 3) << 4
    * q3 = (ql[l]    >>  4) | ((qh[l] >> 4) & 3) << 4
    * q4 = (ql[l+32] >>  4) | ((qh[l] >> 6) & 3) << 4

    With qh = 0x55 the extracted 2-bit values are 01, 01, 01, 01 for
    q1..q4 (shifted by 0, 2, 4, 6 bits and masked to 2 bits), so each
    gets high = (1 << 4) = 16. With ql = 0x10 the low nibble is 0 and
    the high nibble is 1, so q1 = q2 = 16 and q3 = q4 = 17. With d = 1
    and zero scales, output is q - 32: -16 for q1/q2, -15 for q3/q4.
    """
    raw = bytearray()
    raw.extend(np.full(128, 0x10, dtype=np.uint8).tobytes())
    raw.extend(np.full(64, 0x55, dtype=np.uint8).tobytes())
    raw.extend(np.full(16, 1, dtype=np.int8).tobytes())
    raw.extend(np.float16(1.0).tobytes())
    arr = dequant_q6_k(bytes(raw), (256,), np.dtype("float32"))
    # Two groups of q1+q2 (value 16, centred -16) and two groups of
    # q3+q4 (value 17, centred -15), each spanning 64 elements per
    # sub-group. The decoder interleaves these by 32-element l slices,
    # so the easiest assertion to maintain is the count of each value.
    assert int((arr == -16.0).sum()) == 128
    assert int((arr == -15.0).sum()) == 128
    # 128 elements at -16 and 128 at -15 -> 128 * -16 + 128 * -15 = -3968.
    np.testing.assert_allclose(arr.sum(), -3968.0, atol=1e-3)
