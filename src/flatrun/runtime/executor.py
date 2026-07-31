"""Model executor - the bridge between FlatRun and user compute.

The :class:`ModelExecutor` wraps a scheduler + KV cache and exposes a
small, opinionated API for autoregressive decoding. FlatRun's design
goal is *not* to ship a state-of-the-art transformer kernel - it's to
make sure those kernels can run on models that don't fit in RAM.

The executor:

* holds the :class:`LayerScheduler`,
* owns the :class:`KVCache`,
* calls a user-supplied ``forward_fn`` for every layer,
* exposes a clean ``step`` / ``generate`` API for autoregressive use.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from ..utils.errors import ConfigurationError
from ..utils.types import LayerDescriptor
from .kv_cache import KVCache
from .scheduler import LayerHandles, LayerScheduler
from ..core.tensor import TensorHandle


# A forward function takes the loaded handles + the executor's KV cache
# and returns the layer's hidden states / logits.
ForwardFn = Callable[[LayerDescriptor, LayerHandles, KVCache], np.ndarray]


@dataclass(slots=True)
class TokenStep:
    """Result of a single decoder step."""

    layer_outputs: list[np.ndarray]
    last_hidden: np.ndarray
    tokens_processed: int


class ModelExecutor(abc.ABC):
    """Abstract executor interface.

    Concrete implementations live in ``flatrun.model.*``. The runtime
    package ships the abstract type and a reference implementation that
    streams a synthetic forward pass; per-architecture executors (LLama,
    Mistral, ...) can subclass this without changing FlatRun.
    """

    @abc.abstractmethod
    def step(self, tokens: Sequence[int]) -> TokenStep:
        """Run one decoder step over ``tokens``."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release runtime resources."""


class StreamingExecutor(ModelExecutor):
    """Reference executor that streams layers via :class:`LayerScheduler`.

    The class takes a :class:`LayerScheduler` and a :class:`ForwardFn`
    and drives them in lock-step, appending the KV cache after every
    layer. It is intentionally simple - the goal is to demonstrate the
    FlatRun loop, not to compete with optimised transformer kernels.
    """

    def __init__(
        self,
        scheduler: LayerScheduler,
        forward_fn: ForwardFn,
        *,
        kv_cache: KVCache | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._forward = forward_fn
        # Use an explicit None check - ``kv_cache or KVCache()`` is wrong
        # because KVCache defines ``__len__`` and an empty cache is falsy.
        self._kv = kv_cache if kv_cache is not None else KVCache()
        self._kv.ensure_layers(scheduler.layer_count())
        self._closed = False

    @property
    def scheduler(self) -> LayerScheduler:
        return self._scheduler

    @property
    def kv_cache(self) -> KVCache:
        return self._kv

    def step(self, tokens: Sequence[int]) -> TokenStep:
        if self._closed:
            raise ConfigurationError("StreamingExecutor is closed")
        if not tokens:
            raise ConfigurationError("StreamingExecutor.step needs at least one token")

        # Each step is independent: the previous call's K/V entries are
        # discarded so re-feeding the full sequence (as autoregressive
        # generation does) doesn't accumulate. If you want incremental
        # decoding, call :meth:`step_incremental` instead.
        self._kv.reset()

        # Bind the token sequence so the forwarder can pick it up.
        self._scheduler.set_tokens(tokens)
        try:
            outputs: list[np.ndarray] = []
            last_hidden: np.ndarray | None = None

            def per_layer(layer: LayerDescriptor, handles: LayerHandles) -> np.ndarray:
                nonlocal last_hidden
                hidden = self._forward(layer, handles, self._kv)
                last_hidden = hidden
                # Reference attention stub: ensure one K/V entry per layer per
                # step. ``self._forward`` already wrote to the cache for the
                # layers whose forward populates K/V itself; for layers that
                # don't (custom user code), use the last hidden state.
                current = self._kv.stack(layer.index)
                if current is None:
                    self._kv.append(layer.index, hidden, hidden)
                outputs.append(hidden)
                return hidden

            self._scheduler.run(per_layer)
            assert last_hidden is not None
            return TokenStep(
                layer_outputs=outputs,
                last_hidden=last_hidden,
                tokens_processed=len(tokens),
            )
        finally:
            self._scheduler.set_tokens(None)

    def step_incremental(self, tokens: Sequence[int]) -> TokenStep:
        """Like :meth:`step` but keeps K/V entries from prior calls.

        Use this for true autoregressive decoding where each new token
        is appended to the cache rather than re-running the full
        sequence. The current FlatRun forwarder does not yet produce
        ``(seq, hidden)`` -> ``(1, hidden)`` style incremental outputs,
        so this method is provided for forward compatibility.
        """
        # Currently identical to step(); left as a separate hook so
        # future forwarders can opt into incremental K/V updates.
        return self.step(tokens)

    def close(self) -> None:
        if self._closed:
            return
        self._kv.reset()
        self._closed = True


__all__ = [
    "ForwardFn",
    "ModelExecutor",
    "StreamingExecutor",
    "TokenStep",
]