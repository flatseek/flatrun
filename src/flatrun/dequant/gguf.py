"""GGUF quant dequantization.

Pure NumPy implementations for the most common block types in
GGUF files served by LM Studio / llama.cpp:

* ``Q4_0`` - 4-bit symmetric, 32 elements per block.
* ``Q8_0`` - 8-bit symmetric, 32 elements per block.
* ``Q4_K`` - 4-bit K-quant, 256 elements per super-block.
* ``Q5_K`` - 5-bit K-quant, 256 elements per super-block.
* ``Q6_K`` - 6-bit K-quant, 256 elements per super-block.

Every routine is a vectorised transcription of the matching
``dequantize_row_*`` in ``llama.cpp/ggml/src/ggml-quants.c``, and the
element ordering follows it exactly. That ordering is the easy thing to
get wrong: the packed nibbles are *not* interleaved. In Q4_0 the low
nibble of byte ``l`` is element ``l`` and its high nibble is element
``l + 16``; the K-quants use the same split over each 64-element group.
Writing them as ``y[0::2] / y[1::2]`` produces a permuted tensor that
still has the right shape, right dtype and a plausible value
distribution, so it survives every check short of comparing against a
reference decode.
"""

from __future__ import annotations

import numpy as np


def _as_uint8(raw: "bytes | np.ndarray") -> np.ndarray:
    """Wrap a raw byte buffer as a contiguous uint8 ndarray view.

    Both ``bytes`` (the legacy API) and ``np.ndarray`` (zero-copy
    view into the mmap, the new fast path) are accepted. Internal
    callers always receive a uid8 ndarray they can ``reshape`` into
    the per-block layout.
    """
    if isinstance(raw, np.ndarray):
        if raw.dtype == np.uint8:
            return raw
        return raw.astype(np.uint8)
    return np.frombuffer(raw, dtype=np.uint8)


# GGUF is little-endian on the wire regardless of host byte order.
_F16_LE = np.dtype("<f2")

QK_K = 256


def _fp16(raw_arr: np.ndarray, start: int) -> np.ndarray:
    """Read a little-endian fp16 scalar per block as float32."""
    return (
        raw_arr[:, start : start + 2]
        .copy()
        .view(_F16_LE)
        .astype(np.float32)
        .reshape(-1)
    )


def _finish(decoded: np.ndarray, size: int, shape, dtype) -> np.ndarray:
    """Trim block padding and restore the logical shape.

    Pass ``copy=False`` to ``astype`` so the production F32 path
    returns the same buffer instead of allocating a fresh one for
    every call. Without ``copy=False`` the default is ``copy=True``;
    even when the dtype matches the underlying array, NumPy still
    pays for a full F32-to-F32 memcpy just to satisfy the
    "always copy" default. On Qwen3-14B the wasted copy is on the
    order of 365 MB per dequant call (a fully-decoded 5_120×18_944
    tensor), measurable as ~30 ms per call on Apple Silicon.
    """
    return decoded.reshape(-1)[:size].astype(dtype, copy=False).reshape(shape)


# ---------------------------------------------------------------------------
# Q4_0 / Q8_0
# ---------------------------------------------------------------------------


def dequant_q4_0(raw: bytes | np.ndarray, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Decode a ``Q4_0`` tensor.

    Block layout (18 bytes per 32 elements)::

        ggml_half d;
        uint8_t   qs[16];

    Reference ``dequantize_row_q4_0``::

        y[i*qk + j]         = ((qs[j] & 0x0F) - 8) * d
        y[i*qk + j + qk/2]  = ((qs[j] >>   4) - 8) * d

    so the low nibbles fill the first half of the block and the high
    nibbles the second half.
    """
    size = int(np.prod(shape))
    if size == 0:
        return np.empty(0, dtype=dtype).reshape(shape)
    n_blocks = (size + 31) // 32
    raw_arr = _as_uint8(raw).reshape(n_blocks, 18)
    d = _fp16(raw_arr, 0)
    qs = raw_arr[:, 2:]

    decoded = np.empty((n_blocks, 32), dtype=np.float32)
    decoded[:, :16] = (qs & 0x0F).astype(np.float32) - 8.0
    decoded[:, 16:] = (qs >> 4).astype(np.float32) - 8.0
    decoded *= d[:, None]
    return _finish(decoded, size, shape, dtype)


def dequant_q8_0(raw: bytes | np.ndarray, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Decode a ``Q8_0`` tensor.

    Block layout (34 bytes per 32 elements)::

        ggml_half d;
        int8_t    qs[32];
    """
    size = int(np.prod(shape))
    if size == 0:
        return np.empty(0, dtype=dtype).reshape(shape)
    n_blocks = (size + 31) // 32
    raw_arr = _as_uint8(raw).reshape(n_blocks, 34)
    d = _fp16(raw_arr, 0)
    qs = raw_arr[:, 2:].view(np.int8).astype(np.float32)
    return _finish(qs * d[:, None], size, shape, dtype)


# ---------------------------------------------------------------------------
# K-quants
# ---------------------------------------------------------------------------


def _get_scale_min_k4(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack the 12-byte K-quant scale/min block into 8 6-bit pairs.

    Vectorised ``get_scale_min_k4``. The first four sub-blocks store a
    full 6-bit value in one byte; the last four borrow their top two
    bits from the high bits of an earlier byte, which is why reading
    only the low nibble silently halves the scale of the second half of
    every super-block::

        j < 4:  d = q[j] & 63
                m = q[j+4] & 63
        j >= 4: d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
                m = (q[j+4] >>  4) | ((q[j]   >> 6) << 4)

    Returns ``(sc, m)``, each ``(n_blocks, 8)`` float32.
    """
    q = scales.astype(np.uint16)
    n_blocks = q.shape[0]
    sc = np.empty((n_blocks, 8), dtype=np.float32)
    m = np.empty((n_blocks, 8), dtype=np.float32)

    # j = 0..3
    sc[:, :4] = (q[:, 0:4] & 63).astype(np.float32)
    m[:, :4] = (q[:, 4:8] & 63).astype(np.float32)

    # j = 4..7  ->  q[j+4] is q[8:12], q[j-4] is q[0:4], q[j] is q[4:8]
    sc[:, 4:] = ((q[:, 8:12] & 0x0F) | ((q[:, 0:4] >> 6) << 4)).astype(np.float32)
    m[:, 4:] = ((q[:, 8:12] >> 4) | ((q[:, 4:8] >> 6) << 4)).astype(np.float32)
    return sc, m


def dequant_q4_k(raw: bytes, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Decode a ``Q4_K`` tensor.

    Block layout (144 bytes per 256 elements)::

        ggml_half d;          // scale of the quantised scales
        ggml_half dmin;       // scale of the quantised mins
        uint8_t   scales[12]; // 8 x (6-bit scale, 6-bit min)
        uint8_t   qs[128];

    Each 64-element group consumes 32 packed bytes: the low nibbles
    become the first 32 outputs (scale/min pair ``is``) and the high
    nibbles the next 32 (pair ``is + 1``). Values are unsigned 0..15 -
    unlike Q4_0 there is no -8 offset, because the per-sub-block
    minimum is subtracted instead.
    """
    size = int(np.prod(shape))
    if size == 0:
        return np.empty(0, dtype=dtype).reshape(shape)
    n_blocks = (size + QK_K - 1) // QK_K
    raw_arr = _as_uint8(raw).reshape(n_blocks, 144)

    d = _fp16(raw_arr, 0)[:, None]
    dmin = _fp16(raw_arr, 2)[:, None]
    sc, m = _get_scale_min_k4(raw_arr[:, 4:16])
    qs = raw_arr[:, 16:144]

    decoded = np.empty((n_blocks, QK_K), dtype=np.float32)
    for g in range(4):
        chunk = qs[:, g * 32 : (g + 1) * 32]
        lo = g * 64
        decoded[:, lo : lo + 32] = (
            d * sc[:, 2 * g, None] * (chunk & 0x0F).astype(np.float32)
            - dmin * m[:, 2 * g, None]
        )
        decoded[:, lo + 32 : lo + 64] = (
            d * sc[:, 2 * g + 1, None] * (chunk >> 4).astype(np.float32)
            - dmin * m[:, 2 * g + 1, None]
        )
    return _finish(decoded, size, shape, dtype)


def dequant_q5_k(raw: bytes, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Decode a ``Q5_K`` tensor.

    Block layout (176 bytes per 256 elements)::

        ggml_half d;
        ggml_half dmin;
        uint8_t   scales[12];
        uint8_t   qh[32];      // one 5th bit per element
        uint8_t   qs[128];

    ``qh`` precedes ``qs`` on disk, and the same 32 ``qh`` bytes are
    reused by all four groups: group ``g`` tests bit ``2g`` for the low
    nibbles and bit ``2g+1`` for the high ones, contributing 16 to the
    quantised value. Dropping that bit turns Q5_K into a lossy Q4_K.
    """
    size = int(np.prod(shape))
    if size == 0:
        return np.empty(0, dtype=dtype).reshape(shape)
    n_blocks = (size + QK_K - 1) // QK_K
    raw_arr = _as_uint8(raw).reshape(n_blocks, 176)

    d = _fp16(raw_arr, 0)[:, None]
    dmin = _fp16(raw_arr, 2)[:, None]
    sc, m = _get_scale_min_k4(raw_arr[:, 4:16])
    qh = raw_arr[:, 16:48]
    qs = raw_arr[:, 48:176]

    decoded = np.empty((n_blocks, QK_K), dtype=np.float32)
    for g in range(4):
        chunk = qs[:, g * 32 : (g + 1) * 32]
        u1 = np.uint8(1 << (2 * g))
        u2 = np.uint8(2 << (2 * g))
        lo = g * 64
        v1 = (chunk & 0x0F).astype(np.float32) + np.where(qh & u1, 16.0, 0.0)
        v2 = (chunk >> 4).astype(np.float32) + np.where(qh & u2, 16.0, 0.0)
        decoded[:, lo : lo + 32] = d * sc[:, 2 * g, None] * v1 - dmin * m[:, 2 * g, None]
        decoded[:, lo + 32 : lo + 64] = (
            d * sc[:, 2 * g + 1, None] * v2 - dmin * m[:, 2 * g + 1, None]
        )
    return _finish(decoded, size, shape, dtype)


def dequant_q1_0(raw: bytes | np.ndarray, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Decode a ``Q1_0`` (Bonsai 1-bit) tensor.

    Block layout (18 bytes per 128 elements)::

        ggml_half d;
        uint8_t   qs[16];    // 128 bits, 1 per element

    Each bit maps to ``+d`` if set, ``-d`` if not. Effective 1.125 bpw
    (1 sign bit + 16-bit scale amortised over 128 weights). Format
    introduced by the PrismML fork of llama.cpp and not in upstream;
    see https://github.com/PrismML-Eng/llama.cpp.

    `numpy.unpackbits` returns bits in MSB-first order within each byte,
    while the reference C loops over ``bit_offset = j % 8`` - i.e. LSB
    first. Reversing each byte's 8 bits restores the LSB-first
    ordering that the model was trained against.
    """
    size = int(np.prod(shape))
    if size == 0:
        return np.empty(0, dtype=dtype).reshape(shape)
    block_elems = 128
    n_blocks = -(-size // block_elems)  # ceil division; tail bits stay zero
    raw_arr = _as_uint8(raw).reshape(n_blocks, 18)
    d = _fp16(raw_arr, 0)
    qs = raw_arr[:, 2:18]
    # ``np.unpackbits`` is MSB-first within each byte; the reference C
    # loop walks ``bit_offset = j % 8`` (LSB-first). Reverse every 8
    # bits so the layout lines up.
    bits_msb = np.unpackbits(qs, axis=1).reshape(n_blocks, 16, 8)
    bits_lsb = bits_msb[:, :, ::-1].reshape(n_blocks, block_elems)
    sign = np.where(bits_lsb > 0, 1.0, -1.0).astype(np.float32)
    decoded = sign * d[:, None]
    return _finish(decoded, size, shape, dtype)


def dequant_q5_0(raw: bytes | np.ndarray, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Decode a ``Q5_0`` tensor.

    Block layout (22 bytes per 32 elements)::

        ggml_half d;
        uint8_t   qh[4];   // 32 high bits, interleaved across the halves
        uint8_t   qs[16];
    """
    size = int(np.prod(shape))
    if size == 0:
        return np.empty(0, dtype=dtype).reshape(shape)
    n_blocks = (size + 31) // 32
    raw_arr = _as_uint8(raw).reshape(n_blocks, 22)
    d = _fp16(raw_arr, 0)
    qh = raw_arr[:, 2:6].copy().view(np.uint32).reshape(-1)
    qs = raw_arr[:, 6:22]

    j = np.arange(16)
    xh0 = (((qh[:, None] >> (j[None, :] + 0)) << 4) & 0x10).astype(np.float32)
    xh1 = (((qh[:, None] >> (j[None, :] + 12)) & 0x10)).astype(np.float32)
    q0 = ((qs & 0x0F) | xh0.astype(np.uint8)).astype(np.float32) - 16.0
    q1 = ((qs >> 4) | xh1.astype(np.uint8)).astype(np.float32) - 16.0

    decoded = np.empty((n_blocks, 32), dtype=np.float32)
    decoded[:, :16] = q0
    decoded[:, 16:] = q1
    decoded *= d[:, None]
    return _finish(decoded, size, shape, dtype)


def dequant_q5_1(raw: bytes | np.ndarray, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Decode a ``Q5_1`` tensor.

    Block layout (24 bytes per 32 elements)::

        ggml_half d;
        ggml_half m;       // unscaled minimum per block
        uint8_t   qh[4];
        uint8_t   qs[16];
    """
    size = int(np.prod(shape))
    if size == 0:
        return np.empty(0, dtype=dtype).reshape(shape)
    n_blocks = (size + 31) // 32
    raw_arr = _as_uint8(raw).reshape(n_blocks, 24)
    d = _fp16(raw_arr, 0)[:, None]
    m = _fp16(raw_arr, 2)[:, None]
    qh = raw_arr[:, 4:8].copy().view(np.uint32).reshape(-1)
    qs = raw_arr[:, 8:24]

    j = np.arange(16)
    xh0 = (((qh[:, None] >> (j[None, :] + 0)) << 4) & 0x10).astype(np.float32)
    xh1 = (((qh[:, None] >> (j[None, :] + 12)) & 0x10)).astype(np.float32)
    q0 = ((qs & 0x0F) | xh0.astype(np.uint8)).astype(np.float32)
    q1 = ((qs >> 4) | xh1.astype(np.uint8)).astype(np.float32)

    decoded = np.empty((n_blocks, 32), dtype=np.float32)
    decoded[:, :16] = d * q0 + m
    decoded[:, 16:] = d * q1 + m
    return _finish(decoded, size, shape, dtype)


def dequant_q6_k(raw: bytes, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Decode a ``Q6_K`` tensor.

    Block layout (210 bytes per 256 elements)::

        uint8_t   ql[128];     // low 4 bits
        uint8_t   qh[64];      // high 2 bits
        int8_t    scales[16];  // signed, one per 16 elements
        ggml_half d;           // super-block scale

    ``d`` is an fp16 - reading those two bytes as ``int16`` turns a
    scale of 0.5 into 15360 and detonates the whole tensor. ``scales``
    is *signed*. Each 128-element half draws from 64 ``ql`` bytes, 32
    ``qh`` bytes and 8 scales, in the strided pattern below (from
    ``dequantize_row_q6_K``)::

        q1 = (ql[l]    & 0xF) | (((qh[l] >> 0) & 3) << 4)   -> y[l]
        q2 = (ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)   -> y[l+32]
        q3 = (ql[l]    >>  4) | (((qh[l] >> 4) & 3) << 4)   -> y[l+64]
        q4 = (ql[l+32] >>  4) | (((qh[l] >> 6) & 3) << 4)   -> y[l+96]
    """
    size = int(np.prod(shape))
    if size == 0:
        return np.empty(0, dtype=dtype).reshape(shape)
    n_blocks = (size + QK_K - 1) // QK_K
    raw_arr = _as_uint8(raw).reshape(n_blocks, 210)

    ql_all = raw_arr[:, 0:128]
    qh_all = raw_arr[:, 128:192]
    scales = raw_arr[:, 192:208].view(np.int8).astype(np.float32)
    d = _fp16(raw_arr, 208)[:, None]

    l = np.arange(32)
    is_idx = l // 16  # 0 for l<16, 1 otherwise

    decoded = np.empty((n_blocks, QK_K), dtype=np.float32)
    for half in range(2):
        ql = ql_all[:, half * 64 : (half + 1) * 64]
        qh = qh_all[:, half * 32 : (half + 1) * 32]
        sc = scales[:, half * 8 : (half + 1) * 8]
        base = half * 128

        lo = ql[:, l]
        hi = ql[:, l + 32]
        h = qh[:, l]
        for k, (nib, sh, off) in enumerate(
            (
                (lo & 0x0F, 0, 0),
                (hi & 0x0F, 2, 32),
                (lo >> 4, 4, 64),
                (hi >> 4, 6, 96),
            )
        ):
            q = nib.astype(np.float32) + (((h >> sh) & 3) << 4).astype(np.float32) - 32.0
            decoded[:, base + off : base + off + 32] = d * sc[:, is_idx + 2 * k] * q
    return _finish(decoded, size, shape, dtype)


__all__ = [
    "dequant_q1_0",
    "dequant_q4_0",
    "dequant_q4_k",
    "dequant_q5_0",
    "dequant_q5_1",
    "dequant_q5_k",
    "dequant_q6_k",
    "dequant_q8_0",
]
