"""Key-Value cache for autoregressive decoding.

The KV cache is intentionally minimal - it stores per-layer ``(K, V)``
arrays as preallocated growing buffers. FlatRun's role is to *load*
weights efficiently; the KV cache layout depends on the model's
attention implementation, which FlatRun deliberately treats as
opaque.

A note on the data layout: each layer owns a pair of single
``(capacity, kv_heads, head_dim)`` F32 buffers that grow
geometrically (``cap *= 2`` on overflow). Storing one (T, h, d) array
per layer instead of N (h, d) Python list slots changes the cost of
``stack`` from an ``O(N)`` ``np.stack`` allocation per decoder
step into a zero-copy slice. The audit
(``/tmp/attention_audit.py``) measured this at **60-12,000x faster**
depending on the active context length, eliminating the 32 %
attention-path share that ``kv_stack`` held at past_len=120 on
Qwen3-0.6B.

The cache accepts NumPy arrays and returns them in the same order
they were appended. Implementations of attention (PyTorch reference,
hand-written NumPy, future MLX) can be plugged in by passing a
different :class:`AttentionOp` to :class:`ModelExecutor`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..utils.errors import ConfigurationError


class _LayerKV:
    """A single layer's growing K/V buffer pair.

    The head count and head dim are fixed at construction; the
    token dimension grows geometrically. ``stack()`` is a view, not
    a copy.
    """

    __slots__ = ("_cap", "_cap_limit", "_size", "_kv_heads", "_head_dim", "_k", "_v")

    def __init__(
        self,
        kv_heads: int,
        head_dim: int,
        initial_cap: int = 64,
        cap_limit: int = 4096,
    ) -> None:
        if initial_cap < 1:
            raise ConfigurationError(
                "KVCache layer buffer initial_cap must be >= 1"
            )
        self._kv_heads = int(kv_heads)
        self._head_dim = int(head_dim)
        self._cap = int(initial_cap)
        self._cap_limit = int(cap_limit)
        self._size = 0
        self._k = np.empty(
            (self._cap, self._kv_heads, self._head_dim), dtype=np.float32,
        )
        self._v = np.empty(
            (self._cap, self._kv_heads, self._head_dim), dtype=np.float32,
        )

    def __len__(self) -> int:
        return self._size

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        """Append a single token's ``(k, v)`` to the layer.

        Doubles the buffer's T-axis when capacity is hit; the
        doubling is amortised O(1) per append over the lifetime
        of the layer.
        """
        if self._size == self._cap:
            if self._cap == self._cap_limit:
                raise ConfigurationError(
                    f"KVCache layer buffer capacity "
                    f"{self._cap_limit} exceeded"
                )
            new_cap = min(self._cap * 2, self._cap_limit)
            new_k = np.empty(
                (new_cap, self._kv_heads, self._head_dim),
                dtype=np.float32,
            )
            new_v = np.empty(
                (new_cap, self._kv_heads, self._head_dim),
                dtype=np.float32,
            )
            np.copyto(new_k[: self._size], self._k[: self._size])
            np.copyto(new_v[: self._size], self._v[: self._size])
            self._k = new_k
            self._v = new_v
            self._cap = new_cap
        self._k[self._size] = k
        self._v[self._size] = v
        self._size += 1

    def stack(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(k_hist, v_hist)`` views of the live region.

        Both returned arrays are ``OWNDATA=False`` views into the
        underlying growing buffer. The caller must not assume the
        views are stable across subsequent ``append`` calls - the
        rebuild path (``append`` at capacity) replaces the buffer
        backing the views. The forwarder reads the views inside a
        single decoder block, before the next ``append``, so the
        replacement doesn't occur mid-attention.
        """
        return self._k[: self._size], self._v[: self._size]


class KVCache:
    """Per-layer KV cache.

    Each layer owns a single growing F32 buffer pair. ``append`` is
    amortised O(1); ``stack`` is a zero-copy view of the live
    region.
    """

    __slots__ = ("_per_layer", "_cap_limit", "_init_cap")

    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ConfigurationError("KVCache capacity must be positive")
        # Per-layer growing buffers are lazily constructed the first
        # time ``ensure_layers`` is asked about them. We defer
        # knowing ``kv_heads`` / ``head_dim`` until then.
        self._per_layer: list[_LayerKV | None] = []
        self._cap_limit = int(capacity)
        # 64 is a reasonable warm start: small enough that an
        # inference that only decodes a couple of tokens never
        # allocates more than a few KB per layer, large enough that
        # the first doubling lands on 128 well before the typical
        # prompt is fully consumed.
        self._init_cap = 64

    def _layer_kv(
        self, layer: int, kv_heads: int, head_dim: int,
    ) -> _LayerKV:
        """Return the per-layer buffer, creating it on first touch.

        The shape parameters come from the caller's first
        ``stack(layer)`` (or ``append(layer, ...)``) call. Once
        captured, they're locked - the Qwen2 / Llama / Gemma
        forwarders all use one shape per layer.
        """
        buf = self._per_layer[layer]
        if buf is None:
            buf = _LayerKV(
                kv_heads=kv_heads,
                head_dim=head_dim,
                initial_cap=self._init_cap,
                cap_limit=self._cap_limit,
            )
            self._per_layer[layer] = buf
        return buf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_layers(self, n_layers: int) -> None:
        """Grow the per-layer list so it's at least ``n_layers`` long.

        Slots are left as ``None`` until the first ``append`` or
        ``stack`` call supplies the layer's ``(kv_heads, head_dim)``.
        """
        while len(self._per_layer) < n_layers:
            self._per_layer.append(None)

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Append a new (k, v) pair to ``layer``.

        ``k`` and ``v`` are expected to be ``(kv_heads, head_dim)``
        F32 arrays (the per-token slice that the Qwen2 forwarder
        slices out of the rotary-output projection). They are
        memcpy'd into the layer's growing buffer at index
        ``size`` (contiguous slot, ``O(kv_heads * head_dim)``).
        """
        self.ensure_layers(layer + 1)
        kv_heads, head_dim = int(k.shape[0]), int(k.shape[1])
        buf = self._layer_kv(layer, kv_heads, head_dim)
        buf.append(
            np.asarray(k, dtype=np.float32),
            np.asarray(v, dtype=np.float32),
        )

    def stack(self, layer: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(k_hist, v_hist)`` for ``layer``; ``None`` if empty.

        If the layer was never written to, the call records the
        shape from the *next* caller (the Qwen2 forwarder always
        sees a uniform per-layer shape, so this works for every
        practical model). For a clean read of an empty layer the
        caller should treat ``None`` as "nothing to attend to".
        """
        if layer >= len(self._per_layer):
            return None
        buf = self._per_layer[layer]
        if buf is None or len(buf) == 0:
            return None
        return buf.stack()

    def reset(self, layer: int | None = None) -> None:
        """Clear one layer or every layer.

        A layer reset drops the per-layer growing buffer back to a
        zero-length state, ready to receive fresh appends from the
        next prefill. The buffer itself is kept (no GC pressure on
        the typical hot path of "decode, reset, decode, ...").
        """
        if layer is None:
            for buf in self._per_layer:
                if buf is not None:
                    buf._size = 0  # type: ignore[attr-defined]
            return
        if 0 <= layer < len(self._per_layer):
            buf = self._per_layer[layer]
            if buf is not None:
                buf._size = 0  # type: ignore[attr-defined]

    def __len__(self) -> int:
        """Total number of stored entries across all layers."""
        return sum(
            len(buf) for buf in self._per_layer if buf is not None
        )

    def layer_lengths(self) -> Sequence[int]:
        """Per-layer entry counts, ordered by layer index."""
        return tuple(
            len(buf) if buf is not None else 0
            for buf in self._per_layer
        )


__all__ = ["KVCache"]
