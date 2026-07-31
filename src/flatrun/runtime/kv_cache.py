"""Key-Value cache for autoregressive decoding.

The KV cache is intentionally minimal - it stores a list of per-layer
``(K, V)`` arrays. FlatRun's role is to *load* weights efficiently; the
KV cache layout depends on the model's attention implementation, which
FlatRun deliberately treats as opaque.

The cache accepts NumPy arrays and returns them in the same order they
were appended. Implementations of attention (PyTorch reference,
hand-written NumPy, future MLX) can be plugged in by passing a different
:class:`AttentionOp` to :class:`ModelExecutor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..utils.errors import ConfigurationError


@dataclass(slots=True)
class KVEntry:
    """A single (layer, token) KV entry."""

    layer: int
    k: np.ndarray
    v: np.ndarray


class KVCache:
    """Per-layer KV cache.

    The cache is layer-indexed; each layer has its own growing list of
    past K/V blocks. ``append`` is O(1); ``stack`` returns a fresh copy
    of every layer's concatenated history.
    """

    __slots__ = ("_per_layer", "_capacity")

    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ConfigurationError("KVCache capacity must be positive")
        self._capacity = int(capacity)
        # Per-layer list of KVEntry. The outer list is sized lazily.
        self._per_layer: list[list[KVEntry]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_layers(self, n_layers: int) -> None:
        """Grow the cache so it can hold ``n_layers``."""
        while len(self._per_layer) < n_layers:
            self._per_layer.append([])

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Append a new (k, v) pair to ``layer``."""
        self.ensure_layers(layer + 1)
        entries = self._per_layer[layer]
        if len(entries) >= self._capacity:
            raise ConfigurationError(
                f"KVCache capacity {self._capacity} exceeded on layer {layer}"
            )
        entries.append(KVEntry(layer=layer, k=np.asarray(k), v=np.asarray(v)))

    def stack(self, layer: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return concatenated (k, v) for ``layer``; ``None`` if empty."""
        if layer >= len(self._per_layer):
            return None
        entries = self._per_layer[layer]
        if not entries:
            return None
        keys = np.stack([e.k for e in entries], axis=0)
        values = np.stack([e.v for e in entries], axis=0)
        return keys, values

    def reset(self, layer: int | None = None) -> None:
        """Clear one layer or every layer."""
        if layer is None:
            for entries in self._per_layer:
                entries.clear()
            return
        if 0 <= layer < len(self._per_layer):
            self._per_layer[layer].clear()

    def __len__(self) -> int:
        """Total number of stored entries across all layers."""
        return sum(len(entries) for entries in self._per_layer)

    def layer_lengths(self) -> Sequence[int]:
        return tuple(len(entries) for entries in self._per_layer)


__all__ = ["KVCache", "KVEntry"]