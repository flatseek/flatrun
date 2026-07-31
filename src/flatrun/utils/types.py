"""Shared data types used across FlatRun.

All public dataclasses are frozen so they can be safely shared between
threads and between the storage, runtime, and scheduler layers without
defensive copying.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:  # pragma: no cover - 3.10+ is the project floor
    TypeAlias = Any  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Storage identifiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TensorKey:
    """Identity of a tensor inside a model file.

    The backend is responsible for resolving a ``TensorKey`` into an
    offset/size pair; the runtime never looks inside the storage format.
    """

    file: str          # Logical file name relative to the model root.
    name: str          # Tensor name as exposed by the format (e.g. layer name).
    backend: str       # Backend identifier ("safetensors", "gguf", ...).

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.backend}://{self.file}#{self.name}"


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    """Static metadata describing a single tensor.

    Offsets are relative to the file containing the tensor; they are not
    adjusted for the base address of the mmap region because that base is
    determined when the file is opened.
    """

    key: TensorKey
    shape: tuple[int, ...]
    dtype: str                       # Format-native dtype string, e.g. "F32".
    byte_size: int                   # Logical byte size of the tensor payload.
    offset: int                      # Offset inside the underlying file.
    quantization: str | None = None  # Optional quant tag ("Q4_K", ...).

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


# ---------------------------------------------------------------------------
# Cache eviction policy
# ---------------------------------------------------------------------------


class EvictionPolicy(str, Enum):
    """Eviction strategy for the in-memory tensor cache."""

    LRU = "lru"
    FIFO = "fifo"
    INFINITE = "infinite"   # Never evict (useful for benchmarks).


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryStats:
    """Snapshot of memory-manager state.

    Field semantics:
        live_bytes       - Bytes currently mapped into the process.
        peak_bytes       - High-water mark of ``live_bytes``.
        cache_entries    - Number of tensors currently in the cache.
        cache_capacity   - Configured maximum entries.
        mmap_total       - Lifetime count of mmap() calls.
        mmap_failures    - Lifetime count of failed mmap attempts.
        releases_total   - Lifetime count of munmap() / release calls.
        hits / misses    - Cache lookup counters.
    """

    live_bytes: int = 0
    peak_bytes: int = 0
    cache_entries: int = 0
    cache_capacity: int = 0
    mmap_total: int = 0
    mmap_failures: int = 0
    releases_total: int = 0
    hits: int = 0
    misses: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "live_bytes": self.live_bytes,
            "peak_bytes": self.peak_bytes,
            "cache_entries": self.cache_entries,
            "cache_capacity": self.cache_capacity,
            "mmap_total": self.mmap_total,
            "mmap_failures": self.mmap_failures,
            "releases_total": self.releases_total,
            "hits": self.hits,
            "misses": self.misses,
        }


@dataclass(slots=True)
class LayerDescriptor:
    """Description of a single decoder layer.

    A layer groups together the tensors that must be resident at the same
    time during inference. ``tensor_names`` are logical names resolved
    through the model's ``ModelManifest``.
    """

    index: int
    tensor_names: tuple[str, ...]
    # Optional async-prefetch hint, in layer units. ``> 0`` means "kick off
    # prefetch for layer (current + lookahead) once the current one is
    # mapped". The runtime may ignore the hint.
    prefetch_lookahead: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


__all__: list[str] = [
    "TensorKey",
    "TensorMetadata",
    "EvictionPolicy",
    "MemoryStats",
    "LayerDescriptor",
]