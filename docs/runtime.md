# Runtime internals

This document covers the moving parts inside `flatrun.runtime`
and how they cooperate to stream a model through RAM.

## MemoryManager

`flatrun.runtime.memory.MemoryManager` owns the `mmap` /
`munmap` calls. Every `acquire(name)` either returns an
already-mapped handle or mmaps a fresh window of the source
file. When the total resident size exceeds the configured
cap, the manager evicts the least-recently-used handles until
the cap is satisfied.

The cache cap is set in `MemoryConfig.cache_bytes`. The
default is 256 MiB; raise it for warm-cache benchmarking,
lower it to exercise eviction paths in tests.

The default probe is `flatrun.utils.memory.default_probe`,
which uses `psutil` if available and falls back to reading
`/proc/self/status` (Linux) or `resource.getrusage` (macOS).
The probe feeds `MemoryManager.peak_rss`, which the
benchmark uses to validate the "constant RAM regardless of
model size" promise.

When a handle closes (via `MmapTensorHandle.close`), the
underlying mmap region is hinted back to the OS through
`madvise(MADV_DONTNEED)` for tensors >= 1 MiB. The hint
asks the kernel to drop the pages from the page cache so the
RAM can be reused for the next layer. On Windows (where
`madvise` doesn't exist) this is a no-op; the layers still
fit, the OS just keeps the pages around.

## LayerScheduler

`flatrun.runtime.scheduler.LayerScheduler` is the per-step
orchestrator. It walks the layer descriptors in order and,
for each one:

1. Calls the user-supplied `prefetch` hook (default: no-op)
   so callers can warm the next layer's weights asynchronously.
2. Acquires the layer's handles from the memory manager.
3. Invokes the user `compute` callback, which is typically a
   closure over the forwarder.
4. Releases the layer so the memory manager can evict it.

The scheduler is intentionally a single iteration per
`executor.step`. There's no prefetch pipeline, no async
overlap, and no batched layers - keeping it sequential makes
peak RAM strictly bounded by one layer's worth of mmap'd
windows plus whatever the forwarder is holding transiently.

The scheduler exposes its `manager` via a read-only property
so the forwarder can load the post-layer tensors (final
norm + LM head) at every layer when the prediction-evolution
analyzer is enabled. Without this, only the last selected
layer would have access to those tensors.

Layer selection (`--max-layers N`, `--layers LIST`) is
honoured at this layer: the scheduler flags the first
selected layer with `LayerHandles.is_first` and the last
with `LayerHandles.is_last`, so the forwarder knows when to
embed tokens and when to apply the final norm + LM head.

## KVCache

`flatrun.runtime.kv_cache.KVCache` is per-layer. The API:

```python
cache.append(layer, k, v)         # (head_dim,) per call
cache.stack(layer) -> (keys, values)  # concatenated history
cache.reset(layer=None)           # clear one or all layers
```

The cache grows in O(1) per step. It is *not* paged to disk
- if you generate very long sequences, set
`KVCache(capacity=...)` to a value larger than the expected
sequence length to avoid `ConfigurationError` on overflow.

The forwarder uses absolute RoPE positions, so the cache is
safe to keep across multiple `executor.step` calls. The
reference CLI currently re-runs the full prefill each step
(see [Limitations](limitations.md)); true incremental
decode is left as a future optimisation.

## StreamingExecutor

`flatrun.runtime.executor.StreamingExecutor` is the high-
level entry point. It owns the scheduler, the KV cache, and
a user-supplied `ForwardFn`. The `step(tokens)` method:

1. Resets the KV cache (full prefill on every step).
2. Binds the tokens onto the scheduler so the forwarder
   can pick them up on layer 0.
3. Walks the scheduler, calling the forwarder for each
   layer.
4. Returns a `TokenStep` whose `last_hidden` is the final
   layer's output (logits if the forwarder applied the LM
   head, hidden states otherwise).

The executor is intentionally a thin wrapper; the heavy
lifting lives in the forwarder and the scheduler.

## What the runtime *doesn't* do

- No multi-threading inside a single model. FlatRun is
  single-threaded by design - streaming a 70B model on a
  16 GB laptop is already thread-bound on the matmuls.
- No async. Every `step` is synchronous.
- No speculative decoding, beam search, or KV-cache sharing
  between requests.
- No CUDA, Metal, or MLX kernel integration. The reference
  forwarder is pure NumPy.
