"""Layer streaming scheduler.

The scheduler is the orchestra conductor for layer-by-layer inference.
It coordinates three actors:

* the model manifest, which knows which tensors live in each layer,
* the :class:`MemoryManager`, which maps and evicts tensors on demand,
* the user-supplied compute callback, which consumes the loaded layer.

Execution contract::

    scheduler.run(compute) ->
        for layer in model.layers:
            manager.prefetch(layer.prefetch_targets)
            handles = manager.acquire_many(layer.tensor_names)
            compute(layer, handles)
            manager.release_many(layer.tensor_names)
        return result

The contract is deliberately minimal so it can be wrapped by any
backend (HuggingFace, GGUF, future remote).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Iterator, Sequence, TypeVar

from ..utils.errors import LayerStreamingError
from ..utils.types import LayerDescriptor
from .memory import MemoryManager
from ..core.tensor import TensorHandle


R = TypeVar("R")


@dataclass(slots=True)
class SchedulerStats:
    """Per-run counters useful for benchmarks and tracing."""

    layers_executed: int = 0
    tensors_loaded: int = 0
    tensors_released: int = 0
    prefetched: int = 0
    failed: int = 0


# Callback types --------------------------------------------------------------

# ``ComputeFn`` receives the layer and a dict of ready handles; it returns
# whatever the caller considers a useful result (logits, hidden states,
# raw bytes for unit tests, etc.). The function must NOT mutate the
# handle dict; the scheduler owns it.
ComputeFn = Callable[[LayerDescriptor, "LayerHandles"], object]

# ``PrefetchFn`` lets callers hook in async prefetching. FlatRun itself
# uses synchronous prefetch (because the manager is synchronous), but
# the callback signature lets a future async build on top without
# changing the scheduler interface.
PrefetchFn = Callable[[LayerDescriptor], None]


class LayerHandles:
    """Mapping-like wrapper around the live handles for a layer.

    Behaves like a dict but also supports iteration order matching the
    layer's declared tensor order. Accessing a missing key raises
    :class:`LayerStreamingError`.

    The class also carries optional context the scheduler sets before
    calling the user's compute function:

    * ``layer_index`` - the layer's position in the model (0-indexed,
      from the manifest). Stable across custom selections.
    * ``tokens`` - the token sequence for the current decoder step.
      This is what the Qwen2 forwarder reads to look up embeddings.
    * ``is_first`` / ``is_last`` - True for the first/last layer in
      *execution order* (the scheduler's selected subset). The Qwen2
      forwarder uses these to decide when to embed tokens and when
      to apply the final norm + LM head.
    * ``position`` - the 0-indexed position within the selected layer
      subset. Useful for tools that want to track "which selected
      layer is this" without re-deriving it from ``layer_index``.

    These are set after construction by the scheduler (the
    constructor accepts ``None`` for them and the scheduler
    overwrites them before invoking the compute callback).
    """

    __slots__ = ("_handles", "_order", "_closed", "layer_index", "tokens", "is_first", "is_last", "position")

    def __init__(self, handles: dict[str, TensorHandle], order: Sequence[str]) -> None:
        self._handles = dict(handles)
        self._order = tuple(order)
        self._closed = False
        self.layer_index: int | None = None
        self.tokens: Sequence[int] | None = None
        # Position within the scheduler's execution order, not the
        # model's original index. ``is_first`` / ``is_last`` are the
        # scheduler's view of "first/last in the layers we are
        # actually running" — they flip regardless of the original
        # index when the user passes a custom layer selection.
        self.is_first: bool = False
        self.is_last: bool = False
        self.position: int = 0

    def __getitem__(self, name: str) -> TensorHandle:
        return self._handles[name]

    def __contains__(self, name: object) -> bool:
        return name in self._handles

    def __iter__(self) -> Iterator[str]:
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._handles)

    def keys(self):
        return self._handles.keys()

    def values(self):
        return self._handles.values()

    def items(self):
        return self._handles.items()

    def get(self, name: str, default: TensorHandle | None = None) -> TensorHandle | None:
        return self._handles.get(name, default)

    def names_in_order(self) -> tuple[str, ...]:
        return self._order

    def close(self) -> None:
        if self._closed:
            return
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._closed = True


class LayerScheduler(Generic[R]):
    """Stream layers through the memory manager in order.

    Parameters
    ----------
    manager : MemoryManager
        Owns the tensor cache.
    layers : Iterable[LayerDescriptor]
        Logical layers in execution order. Materialised once into a list
        so that ``run`` is deterministic.
    """

    def __init__(
        self,
        manager: MemoryManager,
        layers: Iterable[LayerDescriptor],
        *,
        pre_layer_names: Sequence[str] = (),
        post_layer_names: Sequence[str] = (),
    ) -> None:
        self._manager = manager
        self._layers: list[LayerDescriptor] = list(layers)
        if not self._layers:
            raise LayerStreamingError("LayerScheduler needs at least one layer")
        self._stats = SchedulerStats()
        self._prefetch_hook: PrefetchFn | None = None
        # Optional token sequence attached by StreamingExecutor.step().
        # The forwarder reads it via ``LayerHandles.tokens``.
        self._current_tokens: Sequence[int] | None = None
        # Pre/post-layer tensors injected into the first/last layer's handles.
        self._pre_layer_names: tuple[str, ...] = tuple(pre_layer_names)
        self._post_layer_names: tuple[str, ...] = tuple(post_layer_names)

    def set_tokens(self, tokens: Sequence[int] | None) -> None:
        """Bind a token sequence for the current decoder step.

        Called by :class:`StreamingExecutor.step`. Reset to ``None`` at
        the end of each step so subsequent runs don't see stale tokens.
        """
        self._current_tokens = tokens

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_prefetch_hook(self, hook: PrefetchFn | None) -> None:
        """Install an asynchronous prefetch hook (best effort).

        The hook is called once per layer with the upcoming descriptor.
        A naive implementation can ignore the call; an async build can
        submit prefetch work to a thread pool or asyncio loop.
        """
        self._prefetch_hook = hook

    @property
    def stats(self) -> SchedulerStats:
        return self._stats

    @property
    def manager(self) -> MemoryManager:
        """Expose the underlying :class:`MemoryManager`.

        Useful for diagnostics code (e.g. the prediction-evolution
        analyzer) that needs to load post-layer tensors like
        ``model.norm.weight`` and ``lm_head.weight`` at every
        layer, not just the last one. The forwarder closure
        captures the manager so the per-layer recording can
        request those tensors without going through the handles
        dict.
        """
        return self._manager

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, compute: ComputeFn) -> list[R]:
        """Stream all layers through ``compute``.

        Returns a list with one result per layer, in order. The caller
        is responsible for any aggregation (concatenating hidden states,
        selecting the next-token logits, etc.).
        """
        results: list[R] = []
        for position, layer in enumerate(self._layers):
            try:
                self._maybe_prefetch(layer)
                handles = self._acquire_layer(layer, position)
                handles.tokens = self._current_tokens
                result = compute(layer, handles)
                results.append(result)  # type: ignore[arg-type]
                self._release_layer(handles)
                self._stats.layers_executed += 1
            except Exception:
                self._stats.failed += 1
                raise
        return results

    # ------------------------------------------------------------------
    # Single-step API (used by both ``run`` and external drivers)
    # ------------------------------------------------------------------

    def step(self, index: int, compute: ComputeFn) -> object:
        """Run a single layer by index and return its result."""
        if not (0 <= index < len(self._layers)):
            raise LayerStreamingError(f"Layer index {index} out of range")
        layer = self._layers[index]
        self._maybe_prefetch(layer)
        handles = self._acquire_layer(layer, index)
        try:
            return compute(layer, handles)
        finally:
            self._release_layer(handles)

    def layer_count(self) -> int:
        return len(self._layers)

    def layer(self, index: int) -> LayerDescriptor:
        return self._layers[index]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_prefetch(self, layer: LayerDescriptor) -> None:
        if self._prefetch_hook is not None:
            try:
                self._prefetch_hook(layer)
                self._stats.prefetched += 1
            except Exception:
                # Prefetch failures never abort the run.
                pass

    def _acquire_layer(self, layer: LayerDescriptor, position: int) -> LayerHandles:
        handles: dict[str, TensorHandle] = {}
        # First layer in *execution order* picks up pre-layer tensors
        # (embedding, etc.). Detected by position rather than the
        # original ``layer.index`` so a custom layer selection like
        # ``[1, 3, 4, 6, 7, 8]`` still embeds tokens on its first
        # selected layer.
        is_first = position == 0
        is_last = position == len(self._layers) - 1
        if is_first:
            for name in self._pre_layer_names:
                handles[name] = self._manager.acquire(name)
                self._stats.tensors_loaded += 1
        # Last layer in execution order picks up post-layer tensors
        # (final norm, LM head). Same reasoning — position, not
        # original index.
        if is_last:
            for name in self._post_layer_names:
                handles[name] = self._manager.acquire(name)
                self._stats.tensors_loaded += 1
        for name in layer.tensor_names:
            try:
                handles[name] = self._manager.acquire(name)
                self._stats.tensors_loaded += 1
            except Exception:
                # Roll back partial acquisitions before propagating.
                for acquired in handles.values():
                    try:
                        acquired.close()
                    except Exception:
                        pass
                raise
        order = list(self._pre_layer_names) + list(layer.tensor_names)
        if is_last:
            order += list(self._post_layer_names)
        lh = LayerHandles(handles, order)
        lh.layer_index = layer.index
        lh.is_first = is_first
        lh.is_last = is_last
        lh.position = position
        return lh

    def _release_layer(self, handles: LayerHandles) -> None:
        for name in handles.names_in_order():
            if self._manager.release(name):
                self._stats.tensors_released += 1
        handles.close()


__all__ = [
    "ComputeFn",
    "LayerHandles",
    "LayerScheduler",
    "PrefetchFn",
    "SchedulerStats",
]
