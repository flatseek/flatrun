"""Composite backend that delegates lookups to several shards.

HuggingFace checkpoints are commonly shipped as multiple ``.safetensors``
files. :class:`MultiBackend` glues them together so the runtime sees a
single flat namespace while each individual file is still its own
:class:`StorageBackend`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from ..core.tensor import TensorHandle
from ..utils.errors import TensorNotFoundError
from ..utils.types import TensorKey, TensorMetadata
from .base import StorageBackend


class MultiBackend(StorageBackend):
    """Aggregate several backends under one logical namespace.

    The composite is itself a :class:`StorageBackend` so the runtime
    does not need a separate code path for sharded vs. single-file
    models - it just talks to one backend.
    """

    def __init__(
        self,
        backends: Iterable[StorageBackend],
        name: str = "multi",
    ) -> None:
        self._backends: list[StorageBackend] = list(backends)
        if not self._backends:
            raise ValueError("MultiBackend requires at least one child backend")
        self._name = name
        self._index: dict[str, StorageBackend] = {}

    # ------------------------------------------------------------------
    # StorageBackend API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def root(self) -> Path:
        # Return a virtual root path composed of every child root.
        roots = ", ".join(str(b.root) for b in self._backends)
        return Path(f"<multi:{roots}>")

    def open(self) -> None:
        for b in self._backends:
            b.open()
        self._rebuild_index()

    def close(self) -> None:
        for b in self._backends:
            b.close()
        self._index.clear()

    def list_tensors(self) -> Iterator[TensorKey]:
        for b in self._backends:
            yield from b.list_tensors()

    def get_metadata(self, name: str) -> TensorMetadata:
        backend = self._resolve(name)
        return backend.get_metadata(name)

    def has_tensor(self, name: str) -> bool:
        return name in self._index

    @property
    def supports_mmap(self) -> bool:
        return all(b.supports_mmap for b in self._backends)

    def open_handle(self, name: str) -> TensorHandle:
        backend = self._resolve(name)
        return backend.open_handle(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        self._index = {k.name: b for b in self._backends for k in b.list_tensors()}

    def _resolve(self, name: str) -> StorageBackend:
        # Rebuild lazily if the index is empty (e.g. open() was never called).
        if not self._index:
            self._rebuild_index()
        try:
            return self._index[name]
        except KeyError as exc:  # pragma: no cover - trivial
            raise TensorNotFoundError(name, self._name) from exc

    def shards(self) -> tuple[StorageBackend, ...]:
        """Return the child backends as an immutable tuple."""
        return tuple(self._backends)


__all__ = ["MultiBackend"]