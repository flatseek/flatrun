"""Memory manager - the core of FlatRun.

The :class:`MemoryManager` owns:

* every mmap region currently held by the process,
* an LRU (or FIFO) cache of resolved :class:`TensorHandle` objects,
* statistics and peak-RSS monitoring for benchmarks,
* a pluggable eviction policy.

Design constraints from the spec:

* "mmap tensor regions" - the manager maps bytes lazily via the backend,
* "release mappings" - eviction and explicit :meth:`release` must munmap,
* "optional LRU cache" - enabled by default, configurable,
* "configurable cache size" - in bytes *or* in tensor count,
* "memory statistics" - counters for hits, misses, evictions, live bytes,
* "peak RAM monitoring" - uses :mod:`flatrun.utils.memory`,
* "design for SSD / HTTP / S3" - the cache key is the logical
  :class:`TensorKey`, which doesn't care where the bytes came from.

The manager never parses a model file. It asks the backend for a handle
and decides whether to keep the handle resident.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterator

from ..backend.base import StorageBackend
from ..utils.errors import BackendError, MemoryError_
from ..utils import MemoryProbe, default_probe
from ..utils.types import EvictionPolicy, MemoryStats, TensorKey
from ..core.tensor import TensorHandle


@dataclass(slots=True)
class MemoryConfig:
    """Configuration knobs for :class:`MemoryManager`.

    Attributes
    ----------
    cache_bytes : int | None
        Maximum number of cached bytes. ``None`` disables the byte cap.
    cache_entries : int | None
        Maximum number of cached tensors. ``None`` disables the entry cap.
    policy : EvictionPolicy
        Eviction strategy. ``LRU`` (default), ``FIFO``, or ``INFINITE``.
    probe : MemoryProbe | None
        RSS probe used for peak tracking. Defaults to
        :func:`flatrun.utils.memory.default_probe`.
    """

    cache_bytes: int | None = None
    cache_entries: int | None = 64
    policy: EvictionPolicy = EvictionPolicy.LRU
    probe: MemoryProbe | None = None


class MemoryManager:
    """Cache of :class:`TensorHandle` objects keyed by :class:`TensorKey`.

    The manager is the only object in FlatRun that decides when a tensor
    is mapped into RAM and when it is released. The runtime asks for a
    handle by name; the manager calls the backend if necessary, then
    applies the eviction policy when the cache fills up.
    """

    def __init__(
        self,
        backend: StorageBackend,
        config: MemoryConfig | None = None,
    ) -> None:
        self._backend = backend
        self._config = config or MemoryConfig()
        if self._config.probe is None:
            self._probe: MemoryProbe = default_probe()
        else:
            self._probe = self._config.probe
        # OrderedDict preserves insertion order; for LRU we move-to-end on access.
        self._cache: "OrderedDict[TensorKey, TensorHandle]" = OrderedDict()
        self._live_bytes = 0
        self._stats = MemoryStats(cache_capacity=self._capacity_entries())
        self._lock = threading.RLock()
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def backend(self) -> StorageBackend:
        return self._backend

    @property
    def config(self) -> MemoryConfig:
        return self._config

    def acquire(self, name: str) -> TensorHandle:
        """Return a live :class:`TensorHandle` for ``name``.

        Hot path - called by the scheduler for every tensor that enters
        a layer. Returns the cached handle if one is present.
        """
        if self._closed:
            raise MemoryError_("MemoryManager is closed")
        with self._lock:
            key = self._resolve_key(name)
            handle = self._cache.get(key)
            if handle is not None and not handle.closed:
                self._stats.hits += 1
                if self._config.policy is EvictionPolicy.LRU:
                    self._cache.move_to_end(key)
                self._refresh_peak()
                return handle
            self._stats.misses += 1
            try:
                handle = self._backend.open_handle(name)
            except BackendError:
                self._stats.mmap_failures += 1
                raise
            self._stats.mmap_total += 1
            self._cache[key] = handle
            self._live_bytes += handle.byte_size
            if self._config.policy is EvictionPolicy.LRU:
                self._cache.move_to_end(key)
            self._evict_if_needed()
            self._refresh_stats()
            self._refresh_peak()
            return handle

    def prefetch(self, names: list[str]) -> None:
        """Best-effort warm of the cache.

        Errors are swallowed; prefetch is an optimization and must never
        break the caller's hot path.
        """
        for name in names:
            try:
                self.acquire(name)
            except Exception:
                # Prefetch failures are intentionally non-fatal.
                continue

    def release(self, name: str) -> bool:
        """Drop a single tensor from the cache, closing its handle."""
        with self._lock:
            key = self._resolve_key(name)
            handle = self._cache.pop(key, None)
            if handle is None:
                return False
            self._live_bytes -= handle.byte_size
            if self._live_bytes < 0:
                self._live_bytes = 0
            handle.close()
            self._stats.releases_total += 1
            self._refresh_stats()
            return True

    def release_many(self, names: list[str]) -> int:
        """Drop every tensor in ``names``; returns the count actually released."""
        return sum(self.release(n) for n in names)

    def clear(self) -> None:
        """Drop every cached handle."""
        with self._lock:
            for handle in self._cache.values():
                try:
                    handle.close()
                except Exception:
                    pass
            self._cache.clear()
            self._live_bytes = 0
            self._refresh_stats()

    def close(self) -> None:
        """Drop everything; idempotent."""
        with self._lock:
            self.clear()
            self._closed = True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> MemoryStats:
        with self._lock:
            self._refresh_stats()
            return MemoryStats(**self._stats.snapshot())

    def peak_rss(self) -> int:
        return int(self._probe.peak_rss())

    def rss(self) -> int:
        return int(self._probe.rss())

    def cached_names(self) -> list[str]:
        """Snapshot of the names currently resident (debug / tests)."""
        with self._lock:
            return [k.name for k in self._cache.keys()]

    def __len__(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Iteration support (for debugging / benchmarks)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[TensorHandle]:
        with self._lock:
            return iter(list(self._cache.values()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_key(self, name: str) -> TensorKey:
        # Touching the backend here would be expensive; we rely on the
        # caller passing the same canonical name as stored in the file.
        # We build a TensorKey whose ``file`` is unknown - it never has
        # to round-trip through the backend for cache lookup purposes.
        meta = self._backend.get_metadata(name)
        return meta.key

    def _capacity_entries(self) -> int:
        return int(self._config.cache_entries) if self._config.cache_entries is not None else 0

    def _refresh_stats(self) -> None:
        self._stats.live_bytes = int(self._live_bytes)
        self._stats.cache_entries = len(self._cache)
        self._stats.cache_capacity = self._capacity_entries()

    def _refresh_peak(self) -> None:
        peak = self.peak_rss()
        if peak > self._stats.peak_bytes:
            self._stats.peak_bytes = peak

    def _evict_if_needed(self) -> None:
        """Apply cache size and policy constraints."""
        # Infinite policy: no eviction ever.
        if self._config.policy is EvictionPolicy.INFINITE:
            return
        cap_entries = self._config.cache_entries
        cap_bytes = self._config.cache_bytes
        # Evict entries over the count cap (LRU evicts from the front).
        while cap_entries is not None and len(self._cache) > cap_entries:
            self._evict_one()
        # Evict entries over the byte cap.
        while cap_bytes is not None and self._live_bytes > cap_bytes and self._cache:
            self._evict_one()

    def _evict_one(self) -> None:
        if not self._cache:
            return
        if self._config.policy is EvictionPolicy.FIFO:
            key, handle = self._cache.popitem(last=False)
        else:  # LRU
            key, handle = self._cache.popitem(last=False)
        self._live_bytes -= handle.byte_size
        if self._live_bytes < 0:
            self._live_bytes = 0
        handle.close()
        self._stats.releases_total += 1


__all__ = ["MemoryConfig", "MemoryManager"]