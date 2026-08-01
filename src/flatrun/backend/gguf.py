"""GGUF storage backend.

Implements the GGUF v3 file format used by llama.cpp, Ollama, LM Studio,
and most open-weight LLM distributions. The backend:

* Parses the header (magic, version, tensor count, metadata KV count).
* Walks the metadata KV table, exposing common fields through
  ``backend.metadata``.
* Walks the tensor info table, computing each tensor's byte size from
  the GGML quant type.
* Memory-maps the entire file once on :meth:`open` and returns
  zero-copy :class:`MmapTensorHandle` instances on demand.
* Knows nothing about the runtime - it only exposes :class:`TensorHandle`.

Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

Quant type -> (byte size per block, block size in elements) is the table
that turns ``GGML_TYPE`` ids into a payload size. Block sizes match
``ggml.h`` in the upstream repository. Quant types we don't recognise
are stored as ``Unknown (id=N)`` so callers can still inspect them.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..core.tensor import MmapTensorHandle
from ..utils.errors import BackendError, TensorNotFoundError
from ..utils.types import TensorKey, TensorMetadata
from .base import StorageBackend


_GGUF_MAGIC = b"GGUF"
_GGUF_SUPPORTED_VERSIONS = (2, 3)


# ---------------------------------------------------------------------------
# GGML type table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GGMLTypeSpec:
    """Block layout for a GGML quant type."""

    name: str
    type_size: int         # bytes per block
    block_size: int        # elements per block
    label: str = ""        # alternate label (Q4_K vs q4_K)


# Source: ggml.h in llama.cpp.
_GGML_TYPES: dict[int, GGMLTypeSpec] = {
    0:  GGMLTypeSpec("F32",     4,   1),
    1:  GGMLTypeSpec("F16",     2,   1),
    2:  GGMLTypeSpec("Q4_0",    18,  32),
    3:  GGMLTypeSpec("Q4_1",    20,  32),
    6:  GGMLTypeSpec("Q5_0",    22,  32),
    7:  GGMLTypeSpec("Q5_1",    24,  32),
    8:  GGMLTypeSpec("Q8_0",    34,  32),
    9:  GGMLTypeSpec("Q8_1",    36,  32),
    10: GGMLTypeSpec("Q2_K",    84,  256),
    11: GGMLTypeSpec("Q3_K",    110, 256),
    12: GGMLTypeSpec("Q4_K",    144, 256),
    13: GGMLTypeSpec("Q5_K",    176, 256),
    14: GGMLTypeSpec("Q6_K",    210, 256),
    15: GGMLTypeSpec("Q8_K",    292, 256),
    16: GGMLTypeSpec("IQ2_XXS", 66,  256),
    17: GGMLTypeSpec("IQ2_XS",  82,  256),
    18: GGMLTypeSpec("IQ3_XXS", 98,  256),
    19: GGMLTypeSpec("IQ1_S",   50,  256),
    20: GGMLTypeSpec("IQ4_NL",  136, 256),
    21: GGMLTypeSpec("IQ3_S",   114, 256),
    22: GGMLTypeSpec("IQ2_S",   82,  256),
    23: GGMLTypeSpec("IQ4_XS",  136, 256),
    # Not in upstream llama.cpp; introduced by the PrismML fork for
    # Bonsai-style 1-bit models. Block is 128 elements, 16 bytes of
    # packed bits, plus 2 bytes of fp16 scale.
    41: GGMLTypeSpec("Q1_0",    18,  128),
}


def _ggml_block_bytes(type_id: int, numel: int) -> int:
    """Return the byte size of a tensor with the given GGML type.

    ``numel`` is the logical number of elements (after dequantisation).
    Quant tensors store ``(numel // block_size) * type_size`` bytes.
    """
    spec = _GGML_TYPES.get(type_id)
    if spec is None:
        # Fall back: pretend F16. The handle will be wrong but the error
        # will surface when the user tries to use the data.
        return numel * 2
    if spec.block_size == 1:
        return numel * spec.type_size
    n_blocks = (numel + spec.block_size - 1) // spec.block_size
    return n_blocks * spec.type_size


def _ggml_type_name(type_id: int) -> str:
    spec = _GGML_TYPES.get(type_id)
    if spec is None:
        return f"Unknown({type_id})"
    return spec.name


def _ggml_to_numpy_dtype(ggml_name: str) -> str:
    """Map a GGML type name to a NumPy-compatible dtype string.

    Quant types have no native NumPy representation - they are exposed
    as ``uint8`` byte buffers, and the caller is responsible for
    dequantisation via :mod:`flatrun.dequant` or a custom kernel.
    """
    table = {
        "F32": "float32",
        "F16": "float16",
    }
    return table.get(ggml_name, "uint8")


# ---------------------------------------------------------------------------
# Metadata value types
# ---------------------------------------------------------------------------

_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_INT64 = 10
_GGUF_TYPE_FLOAT64 = 11

_TYPE_STRUCT = {
    _GGUF_TYPE_UINT8: ("<B", 1),
    _GGUF_TYPE_INT8: ("<b", 1),
    _GGUF_TYPE_UINT16: ("<H", 2),
    _GGUF_TYPE_INT16: ("<h", 2),
    _GGUF_TYPE_UINT32: ("<I", 4),
    _GGUF_TYPE_INT32: ("<i", 4),
    _GGUF_TYPE_FLOAT32: ("<f", 4),
    _GGUF_TYPE_BOOL: ("<?", 1),
    _GGUF_TYPE_INT64: ("<q", 8),
    _GGUF_TYPE_FLOAT64: ("<d", 8),
}


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _TensorInfo:
    """Parsed tensor-info entry."""

    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int   # absolute offset into the file (computed from base + relative)
    byte_size: int


# Translation table from GGUF / llama.cpp names to HuggingFace Qwen2 names.
# Used when the manifest builder sets ``gguf_name_translation=True``.
_GGUF_TO_HF_QWEN2 = {
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output": "lm_head",
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def translate_gguf_name(gguf_name: str, architecture: str | None = None) -> str:
    """Translate a GGUF tensor name to HuggingFace Qwen2 naming.

    Example:
        ``blk.0.attn_q.weight`` -> ``model.layers.0.self_attn.q_proj.weight``
    """
    if architecture in (None, "qwen2"):
        # blk.<i>.<rest>
        if gguf_name.startswith("blk."):
            parts = gguf_name.split(".", 2)
            if len(parts) == 3:
                idx, rest = parts[1], parts[2]
                # rest is something like "attn_q.weight" - split off the suffix
                # (.weight, .bias, .biases, .scales).
                for suffix in (".weight", ".bias", ".biases", ".scales"):
                    if rest.endswith(suffix):
                        base = rest[: -len(suffix)]
                        translated = _GGUF_TO_HF_QWEN2.get(base, base)
                        return f"model.layers.{idx}.{translated}{suffix}"
        # No blk.X prefix - look up directly.
        for suffix in (".weight", ".bias", ".biases", ".scales"):
            if gguf_name.endswith(suffix):
                base = gguf_name[: -len(suffix)]
                translated = _GGUF_TO_HF_QWEN2.get(base, base)
                return f"{translated}{suffix}"
    return gguf_name


class GGUFBackend(StorageBackend):
    """Storage backend for GGUF v2 / v3 files.

    Parameters
    ----------
    path : Path | str
        Path to the ``.gguf`` file.
    mmap : bool
        When ``True`` (default) the file is mmapped and handles are
        zero-copy. Set to ``False`` for filesystems that don't support
        mmap.
    """

    def __init__(self, path: Path | str, *, mmap: bool = True, translate_names: bool = True) -> None:
        self._path = Path(path)
        self._use_mmap = bool(mmap)
        self._translate = bool(translate_names)
        self._fd: int | None = None
        self._mmap: mmap.mmap | None = None
        self._metadata: dict[str, TensorMetadata] = {}
        self._raw_metadata: dict[str, object] = {}
        self._tensors: dict[str, _TensorInfo] = {}
        self._data_offset: int = 0
        self._opened = False

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "gguf"

    @property
    def root(self) -> Path:
        return self._path

    @property
    def gguf_metadata(self) -> dict[str, object]:
        """Access the parsed GGUF metadata KV table.

        Values come back in their native Python type (str, int, float,
        bool, list). The keys are the same names as the upstream tool.
        """
        return dict(self._raw_metadata)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._opened:
            return
        if not self._path.is_file():
            raise BackendError(f"GGUF file not found: {self._path}")
        self._fd = os.open(self._path, os.O_RDONLY)
        file_size = os.fstat(self._fd).st_size
        if file_size < 24:
            raise BackendError(f"GGUF file is too small: {self._path}")

        try:
            header = self._parse_header(self._fd, file_size)
        except BackendError:
            os.close(self._fd)
            self._fd = None
            raise

        # Mmap.
        if self._use_mmap:
            try:
                self._mmap = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
            except (OSError, ValueError):
                self._mmap = None
                self._use_mmap = False

        # Tensor info table starts after metadata KVs; data offset is
        # ``header_end + tensor_count * (n_dims * 8 + 12)`` plus alignment.
        # We trust the parser to give us the correct data offset.
        self._data_offset = header["data_offset"]
        tensors = header["tensors"]
        arch = self._raw_metadata.get("general.architecture") if False else None  # filled below
        for info in tensors:
            ggml_type = info["ggml_type"]
            numel = 1
            for d in info["shape"]:
                numel *= int(d)
            byte_size = _ggml_block_bytes(ggml_type, numel)
            offset = self._data_offset + int(info["offset"])
            # Translate the name to HuggingFace conventions when requested.
            raw_name = info["name"]
            display_name = translate_gguf_name(raw_name, arch) if self._translate else raw_name
            ti = _TensorInfo(
                name=display_name,
                shape=tuple(info["shape"]),
                ggml_type=ggml_type,
                offset=offset,
                byte_size=byte_size,
            )
            self._tensors[ti.name] = ti
            meta = TensorMetadata(
                key=TensorKey(file=self._path.name, name=ti.name, backend=self.name),
                shape=ti.shape,
                dtype=_ggml_to_numpy_dtype(_ggml_type_name(ti.ggml_type)),
                byte_size=ti.byte_size,
                offset=ti.offset,
                # Only quantised types have a non-None ``quantization``
                # tag. F32 / F16 are stored element-by-element so the
                # logical shape + dtype match the byte count exactly;
                # setting them to a quant name confuses callers (the
                # dequant hot path treats ``None`` as the "no decoder
                # needed" gate).
                quantization=_ggml_type_name(ti.ggml_type)
                if _ggml_to_numpy_dtype(_ggml_type_name(ti.ggml_type)) == "uint8"
                else None,
            )
            self._metadata[ti.name] = meta

        self._raw_metadata = header["metadata"]
        arch = self._raw_metadata.get("general.architecture")
        # Re-translate now that we know the architecture.
        if self._translate and arch:
            translated_meta: dict[str, TensorMetadata] = {}
            translated_tensors: dict[str, _TensorInfo] = {}
            for original_name, meta in self._metadata.items():
                new_name = translate_gguf_name(original_name, arch)
                if new_name != original_name:
                    # Update metadata with translated name.
                    new_key = TensorKey(file=meta.key.file, name=new_name, backend=meta.key.backend)
                    meta = TensorMetadata(
                        key=new_key,
                        shape=meta.shape,
                        dtype=meta.dtype,
                        byte_size=meta.byte_size,
                        offset=meta.offset,
                        quantization=meta.quantization,
                    )
                translated_meta[new_name] = meta
                if new_name in self._tensors:
                    ti = self._tensors[new_name]
                else:
                    ti = self._tensors.pop(original_name, None)
                if ti is not None:
                    ti = _TensorInfo(
                        name=new_name,
                        shape=ti.shape,
                        ggml_type=ti.ggml_type,
                        offset=ti.offset,
                        byte_size=ti.byte_size,
                    )
                    translated_tensors[new_name] = ti
            self._metadata = translated_meta
            self._tensors = translated_tensors
        self._opened = True

    def close(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        self._metadata.clear()
        self._tensors.clear()
        self._raw_metadata.clear()
        self._opened = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_tensors(self) -> Iterator[TensorKey]:
        self._require_open()
        return iter([m.key for m in self._metadata.values()])

    def get_metadata(self, name: str) -> TensorMetadata:
        self._require_open()
        try:
            return self._metadata[name]
        except KeyError as exc:
            raise TensorNotFoundError(name, self.name) from exc

    def has_tensor(self, name: str) -> bool:
        return name in self._metadata

    @property
    def supports_mmap(self) -> bool:
        return self._use_mmap

    @property
    def byte_size(self) -> int:
        return sum(m.byte_size for m in self._metadata.values())

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def open_handle(self, name: str) -> MmapTensorHandle:
        self._require_open()
        meta = self.get_metadata(name)
        if not self._use_mmap or self._mmap is None:
            data = self._read_at(self._fd, meta.offset, meta.byte_size)
            from ..core.tensor import BufferTensorHandle
            return BufferTensorHandle(meta, data)
        return MmapTensorHandle(meta, self._mmap, offset=meta.offset)

    # ------------------------------------------------------------------
    # Header parser
    # ------------------------------------------------------------------

    def _parse_header(self, fd: int, file_size: int) -> dict[str, Any]:
        # Magic + version + tensor count + metadata count.
        head = self._read_at(fd, 0, 24)
        magic = head[:4]
        if magic != _GGUF_MAGIC:
            raise BackendError(
                f"Bad GGUF magic in {self._path}: expected b'GGUF', got {magic!r}"
            )
        version, tensor_count, metadata_count = struct.unpack("<IQQ", head[4:24])
        if version not in _GGUF_SUPPORTED_VERSIONS:
            raise BackendError(
                f"Unsupported GGUF version {version}; FlatRun supports {_GGUF_SUPPORTED_VERSIONS}"
            )

        cursor = 24
        metadata: dict[str, object] = {}
        for _ in range(metadata_count):
            cursor, key, value = self._read_kv(fd, cursor)
            metadata[key] = value

        # Tensor info table.
        tensors: list[dict[str, Any]] = []
        for _ in range(tensor_count):
            cursor, info = self._read_tensor_info(fd, cursor)
            tensors.append(info)

        # Padding to the alignment boundary. GGUF default is 32 bytes.
        alignment = int(metadata.get("general.alignment", 32))
        # Compute data offset = align_up(cursor, alignment).
        data_offset = (cursor + alignment - 1) // alignment * alignment
        if data_offset > file_size:
            raise BackendError(
                f"Computed GGUF data offset {data_offset} exceeds file size {file_size}"
            )

        return {
            "version": version,
            "tensor_count": tensor_count,
            "metadata": metadata,
            "tensors": tensors,
            "data_offset": data_offset,
        }

    def _read_kv(self, fd: int, cursor: int) -> tuple[int, str, object]:
        # Key: length-prefixed UTF-8 string.
        cursor, key = self._read_string(fd, cursor)
        # Value type.
        type_buf = self._read_at(fd, cursor, 4)
        (value_type,) = struct.unpack("<I", type_buf)
        cursor += 4
        # Value body.
        cursor, value = self._read_kv_value(fd, cursor, value_type)
        return cursor, key, value

    def _read_kv_value(self, fd: int, cursor: int, value_type: int) -> tuple[int, object]:
        if value_type == _GGUF_TYPE_STRING:
            return self._read_string(fd, cursor)
        if value_type == _GGUF_TYPE_ARRAY:
            # Array: element type (uint32) + length (uint64) + items.
            hdr = self._read_at(fd, cursor, 12)
            elem_type, length = struct.unpack("<IQ", hdr)
            cursor += 12
            items: list[object] = []
            for _ in range(length):
                cursor, item = self._read_kv_value(fd, cursor, elem_type)
                items.append(item)
            return cursor, items
        if value_type in _TYPE_STRUCT:
            fmt, sz = _TYPE_STRUCT[value_type]
            buf = self._read_at(fd, cursor, sz)
            (val,) = struct.unpack(fmt, buf)
            cursor += sz
            if value_type == _GGUF_TYPE_BOOL:
                return cursor, bool(val)
            return cursor, val
        raise BackendError(f"Unknown GGUF metadata type id {value_type}")

    def _read_string(self, fd: int, cursor: int) -> tuple[int, str]:
        hdr = self._read_at(fd, cursor, 8)
        (length,) = struct.unpack("<Q", hdr)
        cursor += 8
        if length > 1024 * 1024:
            # Sanity guard - strings shouldn't be that long.
            raise BackendError(f"Suspicious GGUF string length {length}")
        body = self._read_at(fd, cursor, length)
        cursor += length
        # GGUF strings are NUL-terminated; strip the terminator.
        text = body.decode("utf-8", errors="replace")
        if text.endswith("\x00"):
            text = text[:-1]
        return cursor, text

    def _read_tensor_info(self, fd: int, cursor: int) -> tuple[int, dict[str, Any]]:
        cursor, name = self._read_string(fd, cursor)
        # n_dims (uint32)
        buf = self._read_at(fd, cursor, 4)
        (n_dims,) = struct.unpack("<I", buf)
        cursor += 4
        shape: list[int] = []
        for _ in range(n_dims):
            buf = self._read_at(fd, cursor, 8)
            (dim,) = struct.unpack("<Q", buf)
            shape.append(int(dim))
            cursor += 8
        buf = self._read_at(fd, cursor, 12)
        ggml_type, offset = struct.unpack("<IQ", buf)
        cursor += 12
        # GGUF stores dimensions fastest-axis-first (``ne[0]`` is the
        # contiguous one), which is the opposite of NumPy's row-major
        # convention. Reverse them here so ``metadata.shape`` can be
        # handed straight to ``reshape``. Reporting ``ne`` verbatim does
        # not merely transpose a matrix - it reinterprets the same flat
        # buffer under the wrong stride and scrambles the weights, which
        # stays shape-legal all the way to the logits.
        return cursor, {
            "name": name,
            "shape": tuple(reversed(shape)),
            "ggml_type": int(ggml_type),
            "offset": int(offset),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if not self._opened:
            raise BackendError(f"Backend {self.name!r} for {self._path} is not open")

    @staticmethod
    def _read_at(fd: int, offset: int, length: int) -> bytes:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.read(fd, length)


__all__ = ["GGUFBackend", "GGMLTypeSpec", "_GGML_TYPES"]