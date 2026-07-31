"""High-level :class:`InferenceRuntime`.

The runtime is the public entry point of FlatRun. It glues together:

* a :class:`StorageBackend` (the model on disk),
* a :class:`MemoryManager` (the mmap cache),
* a :class:`LayerScheduler` (the streaming loop),
* a :class:`ModelManifest` (the per-model layer layout),
* an :class:`AttentionOp` (the user-supplied attention implementation),
* a :class:`KVCache` (autoregressive state).

The runtime itself contains no model logic - that lives in the
manifest and the user's compute callback. FlatRun's job is to feed
weights to that callback without keeping the entire model in RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ..backend.base import StorageBackend
from ..backend.registry import default_registry
from ..backend.multi import MultiBackend
from ..utils.errors import BackendError, ConfigurationError, FlatRunError
from ..utils.types import LayerDescriptor, TensorKey
from .kv_cache import KVCache
from .memory import MemoryConfig, MemoryManager
from .scheduler import ComputeFn, LayerScheduler, SchedulerStats
from ..core.tensor import TensorHandle


@dataclass(slots=True)
class RuntimeConfig:
    """Top-level runtime configuration.

    Attributes
    ----------
    memory : MemoryConfig
        Cache configuration.
    kv_capacity : int
        Maximum entries per layer in the bundled KV cache.
    parallel_loading : bool
        Reserved for future async backends. Currently unused.
    """

    memory: MemoryConfig | None = None
    kv_capacity: int = 4096
    parallel_loading: bool = False


@dataclass(slots=True)
class RuntimeStats:
    """Aggregated view of scheduler + memory state."""

    scheduler: SchedulerStats
    memory_bytes_live: int
    memory_bytes_peak: int
    rss_current: int
    rss_peak: int


class InferenceRuntime:
    """High-level orchestrator that streams an LLM through RAM.

    Typical use::

        runtime = InferenceRuntime.open("/path/to/model.safetensors")
        scheduler = runtime.build_scheduler(manifest)
        results = scheduler.run(lambda layer, handles: ...)
        runtime.close()

    The runtime never copies weights; the user-supplied ``compute``
    callback decides what to do with them (run a forward pass, count
    bytes, validate ranges, ...).
    """

    def __init__(
        self,
        backend: StorageBackend,
        config: RuntimeConfig | None = None,
    ) -> None:
        self._backend = backend
        self._config = config or RuntimeConfig()
        self._manager = MemoryManager(backend, config=self._config.memory)
        self._kv = KVCache(capacity=self._config.kv_capacity)
        self._closed = False
        # Lazily open the backend if it isn't already.
        try:
            self._backend.open()
        except BackendError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise FlatRunError(f"Failed to open backend: {exc}") from exc

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        path: str | Path | Sequence[str | Path],
        *,
        config: RuntimeConfig | None = None,
        hint: str | None = None,
    ) -> "InferenceRuntime":
        """Open a model file or directory.

        ``path`` may be:

        * a single ``.safetensors`` file,
        * a directory containing one or more ``.safetensors`` files,
        * a sequence of paths - one per shard.

        ``hint`` selects the storage format explicitly. When omitted, the
        path's extension is used.
        """
        registry = default_registry()
        p = Path(path) if not isinstance(path, Sequence) or isinstance(path, str) else None
        if p is None:
            # Sequence of paths - one backend per path.
            backends = [registry.open(Path(x), hint=hint) for x in path]
            backend: StorageBackend = MultiBackend(backends)
        elif p.is_dir():
            shards = sorted(p.glob("*.safetensors"))
            if not shards:
                raise ConfigurationError(
                    f"No .safetensors shards found in directory {p!r}"
                )
            backends = [registry.open(s, hint=hint) for s in shards]
            backend = MultiBackend(backends)
        else:
            backend = registry.open(p, hint=hint)
        return cls(backend, config=config)

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def backend(self) -> StorageBackend:
        return self._backend

    @property
    def manager(self) -> MemoryManager:
        return self._manager

    @property
    def kv(self) -> KVCache:
        return self._kv

    @property
    def config(self) -> RuntimeConfig:
        return self._config

    def list_tensors(self) -> Iterator[TensorKey]:
        return self._backend.list_tensors()

    def has_tensor(self, name: str) -> bool:
        return self._backend.has_tensor(name)

    def get_metadata(self, name: str):
        return self._backend.get_metadata(name)

    def acquire(self, name: str) -> TensorHandle:
        """Resolve a single tensor to a live :class:`TensorHandle`."""
        return self._manager.acquire(name)

    def build_scheduler(
        self,
        layers: Iterable[LayerDescriptor],
        *,
        pre_layer_names: Sequence[str] = (),
        post_layer_names: Sequence[str] = (),
    ) -> LayerScheduler:
        """Construct a :class:`LayerScheduler` bound to this runtime.

        ``pre_layer_names`` are tensors that should be injected into the
        first layer's handle set (typical: embedding). ``post_layer_names``
        are tensors that should be injected into the last layer's handle
        set (typical: final norm + LM head).
        """
        return LayerScheduler(
            self._manager,
            layers,
            pre_layer_names=pre_layer_names,
            post_layer_names=post_layer_names,
        )

    def run(
        self,
        layers: Iterable[LayerDescriptor],
        compute: ComputeFn,
    ) -> list[object]:
        """One-shot helper: build a scheduler and stream every layer."""
        scheduler = self.build_scheduler(layers)
        return scheduler.run(compute)

    def stats(self) -> RuntimeStats:
        mem_stats = self._manager.stats()
        scheduler_stats = self._collect_scheduler_stats()
        return RuntimeStats(
            scheduler=scheduler_stats,
            memory_bytes_live=mem_stats.live_bytes,
            memory_bytes_peak=mem_stats.peak_bytes,
            rss_current=self._manager.rss(),
            rss_peak=self._manager.peak_rss(),
        )

    def _collect_scheduler_stats(self) -> SchedulerStats:
        """Return the most recent scheduler stats, if a run has happened.

        The runtime itself does not retain the scheduler; the latest
        :class:`SchedulerStats` is exposed only while the caller still
        holds the scheduler object. Most users will simply inspect the
        scheduler's ``stats`` attribute directly.
        """
        # Without a live reference we cannot recover the exact counters;
        # return an empty snapshot rather than fabricate numbers.
        return SchedulerStats()

    def close(self) -> None:
        if self._closed:
            return
        self._kv.reset()
        self._manager.close()
        self._backend.close()
        self._closed = True

    def __enter__(self) -> "InferenceRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()


__all__ = ["InferenceRuntime", "RuntimeConfig", "RuntimeStats"]