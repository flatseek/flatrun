"""Tests for the GGUF backend.

The tests build small synthetic GGUF files in-memory and check that the
backend's header parser, tensor name translator, and zero-copy handle
API all behave correctly.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from flatrun.backend.gguf import GGUFBackend, translate_gguf_name
from flatrun.utils.errors import BackendError, TensorNotFoundError


# ---------------------------------------------------------------------------
# Synthetic GGUF writer
# ---------------------------------------------------------------------------


_GGUF_TYPE_STRING = 8
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_INT64 = 10
_GGUF_TYPE_BOOL = 7


def _pack_string(text: str) -> bytes:
    """Length-prefixed NUL-terminated GGUF string."""
    body = (text + "\x00").encode("utf-8")
    return struct.pack("<Q", len(body)) + body


def _pack_kv(key: str, value: object) -> bytes:
    """Encode one metadata KV pair (key + type + value)."""
    out = bytearray()
    out.extend(_pack_string(key))
    if isinstance(value, bool):
        out.extend(struct.pack("<I", _GGUF_TYPE_BOOL))
        out.extend(struct.pack("<?", int(value)))
    elif isinstance(value, int):
        out.extend(struct.pack("<I", _GGUF_TYPE_INT64))
        out.extend(struct.pack("<q", value))
    elif isinstance(value, float):
        out.extend(struct.pack("<I", _GGUF_TYPE_FLOAT32))
        out.extend(struct.pack("<f", value))
    elif isinstance(value, str):
        out.extend(struct.pack("<I", _GGUF_TYPE_STRING))
        out.extend(_pack_string(value))
    else:
        raise TypeError(f"unsupported metadata type {type(value).__name__}")
    return bytes(out)


def _pack_tensor_info(name: str, shape: tuple[int, ...], ggml_type: int, offset: int) -> bytes:
    out = bytearray()
    out.extend(_pack_string(name))
    out.extend(struct.pack("<I", len(shape)))
    for d in shape:
        out.extend(struct.pack("<Q", int(d)))
    out.extend(struct.pack("<I", ggml_type))
    out.extend(struct.pack("<Q", offset))
    return bytes(out)


def _pack_block_bytes(p: int) -> int:
    """Round ``p`` up to the next multiple of 32."""
    return (p + 31) // 32 * 32


_GGML_TYPE_F32 = 0
_GGML_TYPE_F16 = 1
_GGML_TYPE_Q8_0 = 8


def _build_clean_gguf(
    path: Path,
    *,
    metadata: dict[str, object],
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
) -> None:
    """Two-pass GGUF writer.

    Pass 1: write the header + metadata KV table + tensor info records
            recording each info record's byte length. Use zero offsets.
    Pass 2: compute concrete data offsets and the payload positions.
            Rebuild the entire file with the offsets baked in.
    """
    # --- Pass 1: lay out the info-table byte length per tensor ---
    info_lens: list[int] = []
    for name, shape, ggml_type, _payload in tensors:
        info_lens.append(len(_pack_tensor_info(name, shape, ggml_type, 0)))

    # Compute concrete data offsets.
    # Header (24) + metadata length + tensor info lengths (no padding yet).
    header_size = 24
    meta_bytes = b"".join(_pack_kv(k, v) for k, v in metadata.items())
    info_bytes_len = sum(info_lens)
    base = header_size + len(meta_bytes) + info_bytes_len
    # Pad to the next 32-byte boundary for the data section.
    data_start = _pack_block_bytes(base)

    abs_offsets: list[int] = []
    cur = data_start
    for _name, _shape, _ggml_type, payload in tensors:
        abs_offsets.append(cur)
        cur += len(payload)
        # Pad to 32 bytes between tensors for cleanliness.
        cur = _pack_block_bytes(cur)

    # GGUF tensor-info offsets are RELATIVE to ``data_start``. The backend
    # adds the absolute data_offset back when reading.
    rel_offsets = [off - data_start for off in abs_offsets]

    # --- Pass 2: build the final byte string with concrete offsets ---
    out = bytearray()
    out.extend(b"GGUF")
    out.extend(struct.pack("<I", 3))             # version
    out.extend(struct.pack("<Q", len(tensors)))  # tensor_count
    out.extend(struct.pack("<Q", len(metadata))) # metadata_count
    for key, value in metadata.items():
        out.extend(_pack_kv(key, value))
    for (name, shape, ggml_type, _payload), off in zip(tensors, rel_offsets):
        out.extend(_pack_tensor_info(name, shape, ggml_type, off))
    # Sanity: pad to data_start.
    while len(out) < data_start:
        out.append(0)
    for _name, _shape, _ggml_type, payload in tensors:
        out.extend(payload)
        # Pad each tensor to 32 bytes for cleanliness.
        pad = (32 - (len(payload) % 32)) % 32
        out.extend(b"\x00" * pad)

    path.write_bytes(bytes(out))


@pytest.fixture()
def synthetic_gguf(tmp_path: Path) -> Path:
    """A tiny GGUF file with three F32 tensors."""
    metadata = {
        "general.architecture": "qwen2",
        "general.alignment": 32,
        "qwen2.embedding_length": 16,
        "qwen2.block_count": 2,
    }
    tensors = [
        ("blk.0.attn_q.weight", (4, 4), _GGML_TYPE_F32, np.zeros(16, dtype=np.float32).tobytes()),
        ("blk.0.attn_k.weight", (4, 4), _GGML_TYPE_F32, np.arange(16, dtype=np.float32).tobytes()),
        ("blk.0.attn_v.weight", (4, 4), _GGML_TYPE_F32, np.full(16, 7.0, dtype=np.float32).tobytes()),
    ]
    path = tmp_path / "tiny.gguf"
    _build_clean_gguf(path, metadata=metadata, tensors=tensors)
    return path


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def test_translate_gguf_name_blk_prefix() -> None:
    """``blk.0.attn_q.weight`` -> ``model.layers.0.self_attn.q_proj.weight``."""
    out = translate_gguf_name("blk.0.attn_q.weight", "qwen2")
    assert out == "model.layers.0.self_attn.q_proj.weight"


def test_translate_gguf_name_unlayered() -> None:
    """Tensors without the ``blk.X.`` prefix still get their base renamed."""
    out = translate_gguf_name("token_embd.weight", "qwen2")
    assert out == "model.embed_tokens.weight"


def test_translate_gguf_name_passthrough_non_qwen() -> None:
    """Non-qwen2 architectures fall through unchanged."""
    out = translate_gguf_name("blk.0.attn_q.weight", "llama")
    assert out == "blk.0.attn_q.weight"


# ---------------------------------------------------------------------------
# Backend parsing
# ---------------------------------------------------------------------------


def test_gguf_open_lists_and_translates_tensors(synthetic_gguf: Path) -> None:
    backend = GGUFBackend(synthetic_gguf)
    backend.open()
    try:
        names = {k.name for k in backend.list_tensors()}
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "model.layers.0.self_attn.k_proj.weight" in names
        assert "model.layers.0.self_attn.v_proj.weight" in names
    finally:
        backend.close()


def test_gguf_metadata_kv(synthetic_gguf: Path) -> None:
    backend = GGUFBackend(synthetic_gguf)
    backend.open()
    try:
        meta = backend.gguf_metadata
        assert meta["general.architecture"] == "qwen2"
        assert meta["qwen2.embedding_length"] == 16
        assert meta["qwen2.block_count"] == 2
    finally:
        backend.close()


def test_gguf_get_metadata_shape(synthetic_gguf: Path) -> None:
    backend = GGUFBackend(synthetic_gguf)
    backend.open()
    try:
        m = backend.get_metadata("model.layers.0.self_attn.q_proj.weight")
        assert m.shape == (4, 4)
        assert m.quantization == "F32"
        assert m.byte_size == 4 * 4 * 4
        # Offset must be within the file.
        assert m.offset < synthetic_gguf.stat().st_size
    finally:
        backend.close()


def test_gguf_missing_tensor_raises(synthetic_gguf: Path) -> None:
    backend = GGUFBackend(synthetic_gguf)
    backend.open()
    try:
        with pytest.raises(TensorNotFoundError):
            backend.get_metadata("does.not.exist")
    finally:
        backend.close()


def test_gguf_handle_reads_zero_payload(synthetic_gguf: Path) -> None:
    """A F32 zero payload comes back as zeros through the handle."""
    backend = GGUFBackend(synthetic_gguf)
    backend.open()
    try:
        h = backend.open_handle("model.layers.0.self_attn.q_proj.weight")
        try:
            arr = h.as_numpy()
            assert arr.shape == (4, 4)
            np.testing.assert_array_equal(arr, np.zeros((4, 4), dtype=np.float32))
        finally:
            h.close()
    finally:
        backend.close()


def test_gguf_handle_reads_payload(synthetic_gguf: Path) -> None:
    """A tensor with a non-zero payload is read back correctly."""
    backend = GGUFBackend(synthetic_gguf)
    backend.open()
    try:
        h = backend.open_handle("model.layers.0.self_attn.k_proj.weight")
        try:
            arr = h.as_numpy()
            assert arr.shape == (4, 4)
            np.testing.assert_array_equal(arr, np.arange(16, dtype=np.float32).reshape(4, 4))
        finally:
            h.close()
    finally:
        backend.close()


def test_gguf_bad_magic(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"XXXX" + b"\x00" * 64)
    backend = GGUFBackend(bad)
    with pytest.raises(BackendError, match="Bad GGUF magic"):
        backend.open()


def test_gguf_unsupported_version(tmp_path: Path) -> None:
    bad = tmp_path / "bad_version.gguf"
    bad.write_bytes(b"GGUF" + struct.pack("<I", 1) + struct.pack("<QQ", 0, 0))
    backend = GGUFBackend(bad)
    with pytest.raises(BackendError, match="Unsupported GGUF version"):
        backend.open()
