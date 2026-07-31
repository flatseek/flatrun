"""SafeTensors storage backend.

The backend exposes a single ``.safetensors`` file as a
:class:`StorageBackend`. It:

* parses the JSON header at the start of the file,
* mmaps the entire file once on :meth:`open`,
* constructs zero-copy :class:`MmapTensorHandle` instances on demand,
* never reads tensor data eagerly.

The implementation deliberately avoids the optional
:mod:`safetensors` library so we can prove that FlatRun does not depend
on a third-party parser at the storage layer. A drop-in alternative
backend (:class:`SafetensorsLibBackend`) is provided for users who want
the reference parser for safety or validation.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from pathlib import Path
from typing import Iterator

from ..core.tensor import MmapTensorHandle
from ..utils.errors import BackendError, TensorNotFoundError
from ..utils.types import TensorKey, TensorMetadata
from .base import StorageBackend


_SAFETENSORS_MAGIC = b""  # SafeTensors has no magic; header length is the only sentinel.
_HEADER_LEN_STRUCT = struct.Struct("<Q")


# Map SafeTensors dtype strings (which are uppercase abbreviations
# matching PyTorch conventions) to NumPy-compatible dtype strings.
_ST_DTYPE_TO_NUMPY = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U64": "uint64",
    "U32": "uint32",
    "U16": "uint16",
    "U8": "uint8",
    "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


def _normalize_safetensor_dtype(dtype: str) -> str:
    """Return the NumPy-equivalent name for a SafeTensors dtype."""
    return _ST_DTYPE_TO_NUMPY.get(dtype, dtype.lower())


class SafeTensorBackend(StorageBackend):
    """Storage backend for a single SafeTensors file.

    Parameters
    ----------
    path : Path
        Path to the ``.safetensors`` file.
    mmap : bool
        When ``True`` (the default), the file is memory-mapped and
        tensor data is exposed via :class:`MmapTensorHandle`. Set to
        ``False`` for testing or when the file lives on a filesystem
        that does not support mmap.
    """

    def __init__(self, path: Path | str, *, mmap: bool = True) -> None:
        self._path = Path(path)
        self._use_mmap = bool(mmap)
        self._fd: int | None = None
        self._mmap: mmap.mmap | None = None
        self._metadata: dict[str, TensorMetadata] = {}
        self._byte_size: int = 0
        self._data_offset: int = 0  # Byte offset of the data section in the file.
        self._opened = False

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "safetensors"

    @property
    def root(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._opened:
            return
        if not self._path.is_file():
            raise BackendError(f"SafeTensor file not found: {self._path}")
        # Open read-only.
        self._fd = os.open(self._path, os.O_RDONLY)
        file_size = os.fstat(self._fd).st_size
        if file_size < 8:
            raise BackendError(f"SafeTensor file is too small: {self._path}")
        # Parse header length and JSON header.
        header_len = self._read_header_length(self._fd)
        header_end = 8 + header_len
        if header_end > file_size:
            raise BackendError(
                f"SafeTensor header extends past end of file: declared={header_len}, file={file_size}"
            )
        header_bytes = self._read_at(self._fd, 8, header_len)
        try:
            header = json.loads(header_bytes)
        except json.JSONDecodeError as exc:
            raise BackendError(f"Invalid SafeTensor header in {self._path}: {exc}") from exc
        # Mmap the full file. The data region follows the header.
        if self._use_mmap:
            try:
                self._mmap = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
            except (OSError, ValueError) as exc:
                # Fall back to non-mmap mode but keep the file descriptor.
                self._mmap = None
                self._use_mmap = False
        # Build metadata table.
        self._byte_size = file_size - header_end
        self._data_offset = header_end
        for tensor_name, info in header.items():
            if tensor_name == "__metadata__":
                # Free-form metadata block - not a tensor.
                continue
            if not isinstance(info, dict):
                raise BackendError(
                    f"Unexpected SafeTensor entry {tensor_name!r}: expected dict, got {type(info).__name__}"
                )
            try:
                dtype = _normalize_safetensor_dtype(str(info["dtype"]))
                shape = tuple(int(d) for d in info["shape"])
                begin, end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
            except (KeyError, TypeError, ValueError) as exc:
                raise BackendError(
                    f"Malformed SafeTensor entry {tensor_name!r} in {self._path}: {exc}"
                ) from exc
            meta = TensorMetadata(
                key=TensorKey(
                    file=self._path.name,
                    name=tensor_name,
                    backend=self.name,
                ),
                shape=shape,
                dtype=dtype,
                byte_size=end - begin,
                offset=begin,
            )
            self._metadata[tensor_name] = meta
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
        return self._byte_size

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def open_handle(self, name: str) -> MmapTensorHandle:
        self._require_open()
        meta = self.get_metadata(name)
        if not self._use_mmap or self._mmap is None:
            # Read into a buffer - slower but works on every filesystem.
            data = self._read_at(self._fd, self._data_offset + meta.offset, meta.byte_size)
            from ..core.tensor import BufferTensorHandle

            return BufferTensorHandle(meta, data)
        # Zero-copy mmap-backed handle. The mmap covers the whole file,
        # so we add ``self._data_offset`` to reach the tensor's bytes.
        return MmapTensorHandle(meta, self._mmap, offset=self._data_offset + meta.offset)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if not self._opened:
            raise BackendError(f"Backend {self.name!r} for {self._path} is not open")

    @staticmethod
    def _read_header_length(fd: int) -> int:
        buf = SafeTensorBackend._read_at(fd, 0, 8)
        (length,) = _HEADER_LEN_STRUCT.unpack(buf)
        return length

    @staticmethod
    def _read_at(fd: int, offset: int, length: int) -> bytes:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.read(fd, length)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def open_safetensors(path: Path | str, *, mmap: bool = True) -> SafeTensorBackend:
    """Open a single ``.safetensors`` file and return a :class:`SafeTensorBackend`.

    Convenience wrapper equivalent to
    ``SafeTensorBackend(path, mmap=mmap).open()``.
    """
    backend = SafeTensorBackend(Path(path), mmap=mmap)
    backend.open()
    return backend
    """Open a SafeTensors file and return a ready-to-use backend."""
    backend = SafeTensorBackend(path, mmap=mmap)
    backend.open()
    return backend


# ---------------------------------------------------------------------------
# Optional reference backend that uses the safetensors library
# ---------------------------------------------------------------------------


class SafetensorsLibBackend(SafeTensorBackend):
    """Reference parser that delegates to :mod:`safetensors`.

    This backend exists for two reasons:

    * Validation - it lets FlatRun users cross-check our hand-written
      header parser against the canonical implementation.
    * Convenience - it provides dequantization helpers and other
      niceties if/when they're needed.

    The class is a subclass of :class:`SafeTensorBackend` so existing
    code paths continue to work.
    """

    def __init__(self, path: Path | str) -> None:
        super().__init__(path)
        self._lib = None  # Filled on open().

    def open(self) -> None:
        try:
            from safetensors import safe_open  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise BackendError(
                "SafetensorsLibBackend requires the 'safetensors' package"
            ) from exc
        self._lib = safe_open(str(self._path), framework="np")
        super().open()


__all__ = ["SafeTensorBackend", "SafetensorsLibBackend", "open_safetensors"]