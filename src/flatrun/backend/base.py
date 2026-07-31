"""Abstract :class:`StorageBackend` interface.

A :class:`StorageBackend` is the only place in FlatRun that knows how a
model file is laid out on disk. The runtime, scheduler, and tensor layers
must never inspect a SafeTensors header, a GGUF table, or anything else
format-specific. They only call the methods defined here.

The contract is intentionally narrow:

* :meth:`open` / :meth:`close` - lifecycle.
* :meth:`list_tensors` - enumerate every tensor the backend can serve.
* :meth:`get_metadata` - look up a single tensor's static metadata.
* :meth:`open_handle` - return a lightweight :class:`TensorHandle` that
  resolves to either an mmap-backed view, a stream, or an in-RAM array.

The handle is the unit of work for everything downstream. The backend
makes no promises about whether the data is resident after
:meth:`open_handle` returns - that is the memory manager's job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from ..core.tensor import TensorHandle
from ..utils.types import TensorKey, TensorMetadata


class StorageBackend(ABC):
    """Pluggable storage backend abstraction.

    Subclasses describe a single model file or directory tree. The
    runtime combines several backends (one per shard) under a
    :class:`MultiBackend` so a sharded checkpoint can be served without
    the runtime caring which file each tensor lives in.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier such as ``"safetensors"`` or ``"gguf"``."""

    @property
    @abstractmethod
    def root(self) -> Path:
        """Filesystem root this backend was opened from."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self) -> None:
        """Open the underlying file(s) and parse static metadata.

        Must be called before any other method. Safe to call multiple
        times; subsequent calls are no-ops.
        """

    @abstractmethod
    def close(self) -> None:
        """Release all file handles and mappings held by this backend.

        After :meth:`close`, the backend must refuse further reads. The
        memory manager is responsible for releasing any cached handles
        obtained via :meth:`open_handle` before closing.
        """

    def __enter__(self) -> "StorageBackend":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @abstractmethod
    def list_tensors(self) -> Iterator[TensorKey]:
        """Yield every tensor the backend can serve.

        Backends may return the iterator lazily; callers that need a
        concrete list should wrap the result in :class:`list`.
        """

    @abstractmethod
    def get_metadata(self, name: str) -> TensorMetadata:
        """Return static metadata for a single tensor by name."""

    def has_tensor(self, name: str) -> bool:
        """Cheap membership check; defaults to :meth:`list_tensors` scan.

        Backends that can do better (e.g. by consulting a prebuilt
        dictionary) should override this.
        """
        for key in self.list_tensors():
            if key.name == name:
                return True
        return False

    @property
    @abstractmethod
    def supports_mmap(self) -> bool:
        """``True`` when the backend can return zero-copy mmap handles."""

    @property
    def total_byte_size(self) -> int:
        """Sum of all tensor byte sizes - convenience for benchmarks."""
        return sum(self.get_metadata(k.name).byte_size for k in self.list_tensors())

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @abstractmethod
    def open_handle(self, name: str) -> TensorHandle:
        """Return a fresh :class:`TensorHandle` for ``name``.

        The handle is owned by the caller. The backend guarantees that
        the handle's underlying data remains valid until either the
        handle is closed or :meth:`close` is called on the backend.
        """


__all__ = ["StorageBackend"]