# Layer streaming

Layer streaming is the mechanism Flatrun uses to keep its
resident set bounded. The loop, in detail:

```text
step(tokens)
  │
  ├─ scheduler.set_tokens(tokens)
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
tensor is quantised — in which case the bytes are decoded
into a freshly allocated float32 array (the dequant cache in
`flatrun.model.qwen2` keeps the decoded array across steps
to avoid paying the dequant cost twice).

The dequant cache itself lives on the forwarder closure
(`make_qwen2_forwarder`). When enabled
(`--dequant-cache on`, the default), every decoded F32
weight is memoised by on-disk tensor name and reused across
decode steps. With `--dequant-cache-stride N`, the cache is
bounded to the last N layers' worth of weights — useful on
memory-constrained hosts where the unbounded cache would
push RSS past the host limit (a 14 B Q4_K_M model needs ~7
GB of F32 heap with the cache unbound; `--dequant-cache
--dequant-cache-stride 2` keeps ~2 layers' worth resident).
The pre/post-layer bookend tensors (embedding, norm,
lm_head) are auto-immune to the eviction because they carry
no `model.layers.N.` token; they are reused on every step.

## What "release" means

`MemoryManager.release(layer)` does *not* immediately
`munmap` the tensors. It marks the handles as evictable; the
actual `munmap` only happens when the resident set exceeds
the cache cap. This is what makes warm-cache steps cheap:
once a layer has been seen, its handles stick around until
something else pushes them out.

The cap is set in `MemoryConfig.cache_bytes`. For models
where the largest tensor is bigger than the cap, Flatrun
auto-bumps the cap to one tensor's worth and emits a
`bumping cache from X to Y` notice — the alternative would
be refusing to load the model, which is not the spirit of
"RAM-agnostic streaming".

When a handle *does* close (either from eviction or from
`MemoryManager.clear()`), the underlying mmap region is
returned to the OS via `madvise(MADV_DONTNEED)` for tensors
>= 1 MiB. Without this hint, the page cache would keep the
touched mmap pages resident long after the model has moved
on to the next layer — RSS would grow with each layer
touched even though the cache cap is honoured.

## Per-step memory

At any point during a step, the resident set is bounded by:

* the **current layer's** mmap'd tensors (typically the
  largest decoder block of the model),
* the **next layer's** tensors, if the prefetch hook
  successfully warmed it (default: none),
* the **KV cache** (one head_dim float per token per layer
  per KV head, stored in a single growing F32 buffer per
  layer — `kv.stack` is a zero-copy view of the live region),
* the **dequant cache** (one float32 buffer per quantised
  weight, sized to the original weight's element count,
  bounded by `--dequant-cache-stride N` if set),
* the forwarder's transient matmul buffers (now
  zero-allocation for the projection path — the
  `.astype(np.float32, copy=False)` zero-cost dtype
  alignment and the SiLU-fused single F32 buffer mean
  most intermediate buffers are views or in-place
  writes; the remaining transient allocations are the
  `np.concatenate` for `qkv_w` and `gateup_w`).

Peak RSS in the streaming case is approximately
`cache_bytes + KV_cache_bytes + dequant_cache_bytes`.

## Step profiles

The CLI's `--profile` flag prints per-step timing. The first
step is dominated by mmap + dequant (cold cache); subsequent
steps that hit the cache are dominated by the matmuls. A
typical profile on a 0.5 B Q8_0 model with `--dequant-cache
on`:

```text
first step: ~900 ms   (cold dequant + mmap residency)
last step:  ~500 ms   (warm dequant cache, only matmul work)
average:    ~600 ms   (mean over the run)
speedup from warm cache: ~1.8x
```

The ratio depends on the model's quant type and the cache
cap. K-quants need more dequant work than Q8_0; the
unbounded-cache speedup ratio therefore tends to be *larger*
on K-quants (dequant amortises across all decode steps).

With `--dequant-cache off` (pure streaming), every step
re-pays the per-layer dequant cost and the speedup vanishes.
That mode trades RAM for throughput and is the default for
Apple Silicon hosts where the 14 B class F32 cache would
OOM (use `--dequant-cache off` for those, or
`--dequant-cache-stride 2` to bound the cache).

## Where the time goes inside one step

The `--profile-detailed` flag brackets every forward-pass
operation with a microsecond-precision timer and gives a
category-level summary. Audit results (see
[`docs/performance-review.md`](performance-review.md) for the
full breakdown and the audit scripts for reproduction):

| category | typical share | what's in it |
|----------|--------------:|--------------|
| MLP | ~40-45 % | `gateup_proj`, `down_proj`, `silu_mul` (fused in-place) |
| Dequant | ~25-35 % | Q4_K / Q5_K / Q6_K dequant kernel — Python-dispatch bound |
| Attention | ~20-30 % | `qk_matmul`, `av_matmul` (Accelerate sgemm), `rope`, `softmax`, `qk_mask`, `gqa_repeat` |
| Tensor Loading | < 1 % | one mmap + one dequant per layer, both cached after step 0 |
| Norm | ~1-2 % | RMSNorm gains for Q / K / hidden (per-head RMSNorm stays in F32) |

The projection matmul bucket is now ~85-97 % BLAS time on
the per-stage audit. The attention bucket is ~70-85 % BLAS.
The Q4_K dequant kernel itself sits at ~0.9-1.2 GB/s
effective bandwidth — an order of magnitude below the M-class
memory ceiling — because the inner 4-iteration Python loop
and per-iteration ufunc dispatches dominate over the actual
arithmetic. Compiled kernels (Numba / C) are the only path to
a structural speedup on the dequant path.
