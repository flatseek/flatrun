# Layer streaming

Layer streaming is the mechanism FlatRun uses to keep its
resident set bounded. The loop, in detail:

```text
step(tokens)
  │
  ├─ scheduler.reset_tokens()
  │
  ├─ for layer in manifest:
  │     prefetch(layer + 1)                    # user hook, no-op by default
  │     handles = manager.acquire(layer)        # mmaps this layer's tensors
  │     hidden = forwarder(layer, handles, kv)  # runs the decoder block
  │     manager.release(layer)                 # marks for eviction
  │
  └─ return TokenStep(last_hidden=hidden)
```

## What "acquire" means

`MemoryManager.acquire(name)` returns a `TensorHandle` whose
`view()` method hands the caller a zero-copy `np.ndarray`
slice into the source file. No bytes are copied unless the
tensor is quantised - in which case the bytes are decoded
into a freshly allocated float32 array (the dequant cache in
`flatrun.model.qwen2` keeps the decoded array across steps
to avoid paying the dequant cost twice).

## What "release" means

`MemoryManager.release(layer)` does *not* immediately
`munmap` the tensors. It marks the handles as evictable; the
actual `munmap` only happens when the resident set exceeds
the cache cap. This is what makes warm-cache steps cheap:
once a layer has been seen, its handles stick around until
something else pushes them out.

The cap is set in `MemoryConfig.cache_bytes`. For models
where the largest tensor is bigger than the cap, FlatRun
auto-bumps the cap to one tensor's worth and emits a
`bumping cache from X to Y` notice - the alternative would
be refusing to load the model, which is not the spirit of
"RAM-agnostic streaming".

## Per-step memory

At any point during a step, the resident set is bounded by:

* the **current layer's** mmap'd tensors (typically the
  largest decoder block of the model),
* the **next layer's** tensors, if the prefetch hook
  successfully warmed it (default: none),
* the **KV cache** (one head_dim float per token per layer
  per KV head),
* the **dequant cache** (one float32 buffer per quantised
  weight, sized to the original weight's element count),
* and the forwarder's transient matmul buffers, which are
  released as soon as the matmul completes.

Peak RSS in the streaming case is approximately
`cache_bytes + KV_cache_bytes + largest_dequant_buffer`.

## Step profiles

The CLI's `--profile` flag prints per-step timing. The first
step is dominated by mmaps and dequant; subsequent steps
that hit the cache are dominated by the matmuls. A typical
profile on a 0.5 B Q8_0 model:

```text
first step: 3000 ms  (cold cache + dequant)
last step:  500 ms  (warm cache)
average:    800 ms  (mean over the run)
speedup from warm cache: 6.0x
```

The ratio depends on the model's quant type and the cache
cap. K-quants need more dequant work than Q8_0; the speedup
ratio therefore tends to be *larger* on K-quants, because
the dequant amortises across more matmuls.
