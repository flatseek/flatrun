# Changelog

All notable changes to Flatrun are documented in this file. The
format loosely follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [Semantic Versioning](https://semver.org/).

Flatrun has **not** had a public release yet. Every change below
is part of pre-release work toward 0.1.0.

## [Unreleased]

The codebase as it stands today. Read this end-to-end if you're new
to the project — every fix from the audit cycle is listed below in
the order it landed, with the measured impact where the bench
captured it.

### What this version ships

- **Streaming runtime** (`flatrun.runtime`): mmap-based tensor
  access, layer scheduler with LRU eviction, per-layer KV cache
  stored in a preallocated growing F32 buffer, custom memory
  manager with byte cap. On Linux/macOS, mmap'd tensors larger
  than 1 MiB are released back to the OS via
  `madvise(MADV_DONTNEED)` when the handle closes, so RSS stays
  bounded by the per-layer working set rather than the cumulative
  touched-mmap.
- **Storage backends** (`flatrun.backend`):
  - `SafeTensorBackend` — hand-written parser, no third-party deps.
  - `GGUFBackend` — parses GGUF v3 in pure Python, including
    on-disk dequantisation for the common block types.
  - `MultiBackend` — composite backend for sharded checkpoints.
- **Reference forwarder** (`flatrun.model.qwen2`): pure NumPy
  Qwen2 / Llama / Qwen3 / Qwen3.5 / Gemma 3 decoder with
  per-head Q/K RMSNorm (Qwen3/Qwen3.5), NEOX and NORM RoPE
  layouts picked automatically from the GGUF architecture,
  tied or untied LM head, grouped-query attention, bfloat16
  / float16 / float32 weight dtypes. Implements pure-NumPy
  Q4_K / Q5_K / Q6_K / Q8_0 / Q4_0 / Q5_0 / Q5_1 / Q1_0
  dequant (`flatrun.dequant.gguf`) and the MLX 4-bit
  (`weight` + `scales` + `biases`) triple decode
  (`flatrun.dequant.mlx`).
- **Layer selection** (`--max-layers N`, `--layers LIST`):
  run a truncated-depth copy of the model or a custom subset
  of decoder layers in any order with inclusive ranges
  (`--layers 0-6,19-24,34-39`). The scheduler attaches the
  embedding to the first selected layer and the final norm
  + LM head to the last selected layer so the subset is a
  self-contained forward pass. Useful for adaptive inference
  and selective-layer-execution research.
- **Per-token debug table** (`--debug`): per-layer stats for
  every token — `norm`, `delta`, `stable`, `influence`,
  `entropy`, `confidence`, `rank_by_norm`, `rank_delta`,
  `rank_stable`. Special tokens are filtered out by default
  (`--debug-include-special` to opt back in).
- **Prediction Evolution analyzer** (`--debug`): runs the
  final norm + LM head at every layer and tracks the
  next-token prediction. Post-inference summary covers
  per-layer table, confidence growth, most influential
  layers (top-k by `delta_confidence`), prediction changes,
  prediction stabilization layer, and a suggested early
  exit (top1 stable + confidence ≥ 95 % of final +
  entropy/margin within tolerance). JSON via
  `--debug-save-analysis PATH`. The previous `LayerAnalyzer`
  was removed because it scored hidden-state activity that
  didn't track the model's actual decision; the
  `PredictionAnalyzer` measures the next-token decision
  directly.
- **Detailed profiler** (`--profile-detailed`): microsecond
  breakdown of every forward-pass operation (RMSNorm, QKV
  projection, QK matmul, softmax, AV matmul, MLP, ...) per
  layer, plus a percentage summary by category (Attention,
  MLP, Tensor Loading, Dequantization, Norm, Sampling,
  Residual, Other). JSON via `--profile-save PATH`.
- **Dequant cache** (`--dequant-cache on` by default): the
  F32 decoder outputs are memoised by on-disk tensor name
  so the second-and-later decode steps don't redo the
  dequant work. Disable with `--dequant-cache off` for
  pure streaming; bound the cache size with
  `--dequant-cache-stride N` so the Python heap stays
  bounded on 14 B+ models (the unbounded cache adds
  ~7 GB of F32 heap to the resident set on a Qwen3-14B
  Q4_K_M, which OOMs hosts with limited RAM — the
  `--dequant-cache-stride 2` mode keeps current + next
  layer).
- **BPE tokenizer** (`flatrun.tokenizer`): byte-level BPE
  that accepts a HuggingFace `tokenizer.json` (with
  `vocab.json` + `merges.txt` fallback) and reads the
  embedded vocab + merges from a GGUF's `tokenizer.ggml.*`
  block.
- **Reference Qwen3.5 forward path**: full-attention layers
  routed through the same decoder block as Qwen3 with
  per-head Q/K norm (with the `(1 + weight)` gain mode when
  the arch config says so) and the optional output-gate
  matmul (`self_attn.gate_proj`) when the weight is
  present. Linear / DeltaNet layers raise a clear
  `NotImplementedError` rather than producing garbage.
- **Reference Gemma 3 forward path**: per-MLP RMSNorm with
  the `(1 + weight)` gain, gated attention, qk-norm gain,
  pre- and post-feedforward RMSNorm positions, attention
  logit soft-cap. Sliding-window attention wired when the
  arch config sets `sliding_window`. MLX-4bit only.

### Audit-driven fixes — what was found and what changed

Every code change in this version traces back to a
`--profile-detailed` measurement or a synthetic benchmark.
The narrative is intentional: the runtime looks the way
it does because each non-obvious line in the forwarder
exists to fix something the profiler caught.

The audit chain (one session per item) is:

1. **`madvise(MADV_DONTNEED)` on tensor close** for tensors
   >= 1 MiB. Pages leave the page cache after the layer's
   compute, so RSS of streamed large models stays bounded
   by the per-layer working set rather than the cumulative
   touched-mmap. Without this hint the cumulative touched
   pages would inflate RSS until the OS reclaimed them
   under memory pressure, defeating the "constant RAM"
   claim on long sequences.
2. **`--dequant-cache on` as the default**. The previous
   default (`off`) made every decode step re-dequantise
   every layer. The detailed profiler showed the
   dequantisation share at ~74 % of total runtime on
   Qwen3-0.6B Q4_K_M. Flipping the default to `on` drops
   the dequantisation share to single-digit percentages
   after the first step. Memory tradeoff: ~1.5 GB extra
   F32 heap for 0.6B, ~7 GB for 14 B (the audit
   documented this in the CLI help text, since 14 B
   hosts may need `--dequant-cache off` to fit in RAM).
3. **Sliding-window bound on the dequant cache**
   (`_SlidingDequantCache`). The unbounded cache holds
   every decoded weight for the process lifetime, which
   on a 14 B Q4_K_M model needs ~7 GB of F32 heap. The
   sliding cache evicts layer-indexed tensors (parsed from
   the `model.layers.N.` or `language_model.model.layers.N.`
   keys) when the forwarder enters `N + stride` layers
   past them; pre/post-layer tensors (`model.embed_tokens`,
   `model.norm`, `lm_head`) are auto-immune because they
   carry no `layers.N.` token. Five regression tests cover
   the dict interface, unbounded no-eviction behaviour,
   layer-rotation eviction, negative-bound safety, and
   `total_bytes()` accounting. Wire-up via
   `--dequant-cache-stride N`. The forwarder's three
   decoder blocks (`_decoder_block`,
   `_gemma3_decoder_block`, `_qwen35_full_attention_block`)
   call `dequant_cache.enter_layer(idx)` at function entry.
4. **Drop redundant `.astype(np.float32)` in projection
   matmul hot path**. The previous code wrapped every
   weight in `.astype(np.float32)` before the matmul;
   `astype` defaults to `copy=True`, so even when the
   weight was already F32 it allocated a fresh buffer
   (measured 107 ms per `gate_w.astype(F32)` call on
   Qwen3-14B Q4_K_M). Replacing with
   `.astype(np_dtype, copy=False)` at 14 hot-path sites
   (qkv/o/gateup/down across all three decoder blocks,
   plus the LM head and `_compute_layer_logits`) gave
   per-layer projection work a **4.16× speedup** in a
   synthetic benchmark on 14 B shapes; end-to-end on
   Qwen3-0.6B Q4_K_M the top-4 projection time dropped
   from ~10.77 s to ~7.53 s (~30 %).
5. **Drop redundant `.astype(np.float32)` in `_finish`**
   (the shared tail of every GGUF dequant function):
   same `copy=True` issue. Single one-line change lifts
   every dequant (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q5_0,
   Q5_1, Q1_0). On Qwen3-14B Q4_K_M dequant this saves
   ~30 ms per call (~365 MB of F32 memcpy avoided).
6. **Fuse SiLU(gate)·up into a single F32 buffer** via
   an in-place `np.exp` → `+ 1.0` → `np.reciprocal` →
   `np.multiply(_, gate)` → `np.multiply(_, up)` chain.
   The previous code issued five separate Python-level
   NumPy expressions (clip, neg, exp, divide, multiply) and
   allocated five intermediate F32 buffers; the fused
   chain allocates one. Measured **1.40-1.56× speedup**
   on the post-gateup stage. Applied to all three decoder
   blocks. The clip is dropped safely: for
   `silu(x) = x · sigmoid(x)` the IEEE-754 limits give the
   correct asymptotics at ±88 (`sigmoid(±88)` rounds to
   1 / 0 so `silu(±88) → ±x` / `0`), so the clip was a
   safety belt that didn't change the answer up to F32
   round-off.
7. **Fix float64 leak in attention scale multiplication.**
   The production code computed
   `scale = 1.0 / np.sqrt(head_dim)`, which is a Python
   float (float64). When the F32 attention tensor is
   multiplied by a Python float, NEP-50 promotes the
   result to float64 — silently leaking F64 through the
   rest of the attention path: `qk_matmul` output,
   `causal_mask` add, softmax intermediates, `av_matmul`
   output, the post-attention reshape, and **the residual
   `hidden = residual + attn_out`** which became F64 —
   meaning every MLP weight matmul downstream ran on F64
   hidden state and produced F64 output that propagated
   further. Fix: cast `scale` to `np.float32` at the top
   of each decoder block (qwen2/qwen3, gemma3, qwen3.5)
   and in the final norm + LM head path. The
   per-stage audit confirmed the dtype leakage end-to-end
   (attn intermediates dropped from F64 15 KB to F32 7.6
   KB).
8. **Rewrite the KV cache as a per-layer preallocated
   growing F32 buffer** (`_LayerKV`). The previous
   implementation stored past K/V as a Python list of
   per-token `(h, d)` arrays and rebuilt the full
   history via `np.stack` on every `stack(layer)` call.
   The cost was `O(T)` per decoder step — both an
   allocation of a fresh `(T, kv_heads, head_dim)` buffer
   and a copy of every entry's data. On Qwen3-0.6B at
   `past_len=120` this was 32 % of attention runtime.
   The new design folds one (T, h, d) buffer pair per
   layer; `append` writes the next slot, `stack` is a
   view of the live region. Measured isolated speedup on
   `stack()` alone ranged **61× (past=16) to 12,700×
   (past=4096)** — eliminating the 32 % attention share
   that `kv_stack` held at past=120. End-to-end attention
   share fell from ~36 % to ~27 % across three runs on
   Qwen3-0.6B Q4_K_M with `--max-new 10`.

### Architecture invariants the audit enforced

- **All weight transposes are zero-copy views.** `weight.T`
  is a strided view (F-contig flags True, OWNDATA False).
  BLAS handles `transB=T` natively via the cblas cblas
  `cblas_sgemm` dispatch; no `arr.T.copy()` is ever called.
- **All weight slices are zero-copy views.** `qkv[:, :q_dim]`,
  `gate[:, :inter]`, etc. are views; `.reshape(...)` on a
  contiguous slice is itself a view.
- **All output buffers are F32 contiguous.** The BLAS
  reduction side reads from a C-contig `(seq, out)` F32
  buffer — the canonical layout for the cblas `cblas_sgemm`
  output.
- **No F64 leak across the residual boundary.**
  Confirmed by the attention audit and the production fix.

### What's at the pure-NumPy ceiling and what isn't

A series of performance audits (the audit scripts live in
`/tmp/`, see `docs/performance-review.md` for the
narrative of the projection audit) sized each part of the
forward pass. The headline:

- Projection matmul: **85-97 % BLAS time** in zero-overhead
  micro-benchmarks. At the pure-NumPy ceiling.
- Attention: ~**72-85 % BLAS** (qk + av einsums dispatch
  via `np.matmul` to Accelerate sgemm; the remaining
  overhead is RoPE, RMSNorm, causal mask, softmax,
  GQA repeat — each bounded by either NumPy dispatch
  cost or unavoidable allocation). The two easy
  pure-NumPy items remaining are (a) caching the
  `qkv_w` and `gateup_w` `np.concatenate`s so they don't
  re-allocate per decoder step (~8-12 % total runtime
  expected) and (b) replacing `gqa_repeat` `np.repeat`
  with a strided broadcast view (depends on Accelerate's
  acceptance of strided K_full; ~5-10 %).
- Dequantisation on K-quants: **0.9-1.2 GB/s effective
  bandwidth**, an order of magnitude below the 30-40
  GB/s ceiling of Apple M-series unified memory. The
  Q4_K dequant is **Python-dispatch bound, not
  memory-bandwidth bound** — the 4-iteration Python loop
  and 8 ufunc dispatches per call dominate over the
  actual numeric work. The audit from this version
  confirmed that fully vectorising the loop in pure
  NumPy does **not** help (the bench measured 0.7-0.8×
  the loop's throughput because the intermediate F32
  buffers cost more memory bandwidth than they save in
  dispatch). Real K-quant acceleration requires a
  compiled kernel (Numba / Cython / C / vDSP) — out of
  scope for the pure-NumPy runtime.
- KV cache stack: **at the ceiling** (zero-copy view).
- SiLU chain: **at the ceiling** (single F32 buffer).

**Realistic pure-NumPy remaining ceiling: ~15-20 %** of
total runtime, entirely from the two cache-extension
items above (concat + GQA view). Beyond that, only a
compiled kernel rewrite unlocks a structural 3-10×
speedup.

The full audit chain (per-projection, per-attention,
per-dequant sub-stage breakdowns with bench numbers) is
in `docs/performance-review.md`.

### Limitations (the same list lives in `docs/limitations.md`)

- **CPU-only.** No CUDA, no Metal kernel, no MLX decode.
  The forwarder dispatches to Accelerate / OpenBLAS via
  NumPy's matmul; there's no GPU path.
- **Pure NumPy forward.** A 0.5 B Q8_0 model generates
  around 25 tok/s on a single thread of a modern laptop
  CPU. A Qwen3-14B Q4_K_M streams at ~1.5 tok/s on the
  same hardware. Replacing the per-layer matmuls with a
  hand-tuned kernel is the obvious optimisation path;
  the streaming layer (mmap, scheduler, KV cache) is not
  on the critical path.
- **BPE tokenizer only.** SentencePiece (LLaMA-1) and
  WordPiece (BERT) are not implemented.
- **Some chat templates fall back to a generic Qwen2
  default.** Exotic templates (function-calling,
  multi-system) need the Jinja subset to grow.
- **The CLI re-encodes the full prefill on every step** —
  the runtime currently has no incremental decode
  path. The detailed profiler makes this visible
  (`--profile-detailed`).
- **Some larger models (~7 B+) currently produce
  unstable activations or incorrect logits** on hidden
  states past the first ~6 layers (Qwen3-14B was the
  trouble zone during testing). Use `--debug` to see the
  per-layer mean / std / L2 norm / position cosine /
  adjacent-row diff / NaN-Inf count and pin down where
  the drift starts; file an issue with that layer number
  and the model hash so a maintainer can fix it.

### Removed since the early prototype

- **`LayerAnalyzer` (hidden-state scoring)**. Replaced by
  `PredictionAnalyzer` because the previous scoring was
  based on residual-stream activity that scored the
  final layer high simply because of the projection,
  and the suggested subsets contradicted the early-exit
  suggestion. The `PredictionAnalyzer` measures the
  model's actual next-token decision rather than the
  residual-stream movement.

---

## How to read this changelog if you're new

The codebase passed through several audit cycles before
this release. Each cycle identifies the *specific*
bottleneck at that moment, fixes it, and re-measures.
The audit trails (each item above) document:

- **What was slow** (the profile observation).
- **Why** (the audit's root-cause analysis).
- **What changed** (the one or two-file diff).
- **Measured impact** (where captured: ~30 % projection
  time reduction, attention share −9 pp, etc.).
- **What still doesn't help** (e.g., pure-NumPy Q4_K
  vectorisation measured 0.7-0.8× the loop).

The `docs/performance-review.md` companion shows the
audit methodology and the detailed per-stage numbers.
The audit scripts (`/tmp/projection_audit.py`,
`/tmp/attention_audit.py`, `/tmp/dequant_audit.py`) are
reproducible against any GGUF / SafeTensors model the
runtime supports.

## Historical notes

- The streaming executor + memory manager + scheduler
  + KV cache + GGUF backend are the original spine from
  the early development; they haven't had public callers
  beyond the in-tree tests.
- The Qwen2 / Llama reference forwarder was the
  prototype's only compute path. Qwen3 / Qwen3.5 / Gemma
  3 support was added incrementally.
- The `madvise(MADV_DONTNEED)` hint was added after a
  test on a 14 B model showed RSS growing past 30 GB even
  with the cache cap honoured — the page cache was
  holding the touched mmap pages forever.
