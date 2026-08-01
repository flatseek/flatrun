# Changelog

All notable changes to FlatRun are documented in this file. The
format is loosely [Keep a Changelog](https://keepachangelog.com/);
versions follow [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-08-01

### Added

- **Layer selection** (`--max-layers N`, `--layers LIST`): run a
  truncated-depth copy of the model or a custom subset of decoder
  layers in any order with inclusive ranges
  (`--layers 0-6,19-24,34-39`). The scheduler attaches the
  embedding to the first selected layer and the final norm + LM
  head to the last selected layer. Useful for adaptive inference
  and selective-layer-execution research.
- **Per-token debug table** (`--debug`): per-layer stats for every
  token — `norm`, `delta`, `stable`, `influence`, `entropy`,
  `confidence`, `rank_by_norm`, `rank_delta`, `rank_stable`.
  Special tokens are filtered by default
  (`--debug-include-special` to opt back in).
- **Prediction Evolution analyzer** (`--debug`): runs the final
  norm + LM head at every layer and tracks the next-token
  prediction. Post-inference summary covers per-layer table,
  confidence growth, most influential layers (top-k by
  `delta_confidence`), prediction changes, prediction
  stabilization layer, and a suggested early-exit layer
  (top1 stable + confidence >= 95% of final + entropy/margin
  within tolerance). JSON dump via `--debug-save-analysis PATH`.
- **Detailed profiler** (`--profile-detailed`): microsecond
  breakdown of every forward-pass operation (RMSNorm, QKV
  projection, QK matmul, softmax, AV matmul, MLP, ...) per
  layer, plus a percentage summary by category. JSON via
  `--profile-save PATH`. The accompanying
  [performance-review.md](docs/performance-review.md) walks
  through the bottlenecks the profiling revealed.
- **`madvise(MADV_DONTNEED)` on `MmapTensorHandle.close`** for
  tensors >= 1 MiB. Pages leave the page cache after the layer's
  compute, so RSS of streamed large models stays bounded by the
  per-layer working set rather than the cumulative touched-mmap.
- **Qwen3.5** support (full attention path). Linear (DeltaNet)
  layers raise a clear `NotImplementedError` rather than producing
  garbage.
- **Gemma 3** support (MLX-4bit). Per-MLP RMSNorm, gated
  attention, qk-norm gain all wired correctly.
- **`--dequant-cache` defaults to `on`**. The CLI now keeps
  dequantized F32 weights in the Python heap for the process
  lifetime so the second-and-later decode steps don't redo the
  dequant work. On a Qwen3-0.6B Q4_K_M model this drops the
  detailed profile's "Dequantization" share from ~74% to a
  single-digit percentage after the first step. Memory
  tradeoff: ~1.5 GB extra heap for 0.6B, ~7 GB for 14B. Pass
  `--dequant-cache off` to opt back into pure streaming mode
  on memory-constrained hosts or large models.
- **`--dequant-cache-stride N`** bounds the dequant cache to
  the last N layers' worth of weights. Default ``None`` is
  unbounded (every decoded weight held for the process
  lifetime - current behaviour). Smaller N bounds Python
  heap (e.g. ``--dequant-cache-stride 2`` keeps current +
  next layer) at the cost of re-dequantising evicted layers
  at the start of every decode step. Useful on memory-
  constrained hosts where the unbounded cache would OOM.

### Performance findings

The detailed profiler confirmed the suspicion that the largest
specific cost in the forward pass is the **QKV / MLP concat +
astype** pattern: per layer, the forwarder allocates ~360 MB
(qkv) and ~740 MB (gateup) via `np.concatenate` + `.astype(
np.float32)`, even when the matmul itself is the same size.
The second-largest cost is the **autoregressive decode loop**
which re-encodes the entire prompt per step. See
`docs/performance-review.md` for the full analysis.

### Performance fixes (2026-08-01)

- **Redundant `.astype(np.float32)` calls removed from the
  projection matmul hot path** (`qwen2.py:1160`, `1256`,
  `1289`, `~1287`, `1391`, `1445`, `1469`, `1473`, `1537`,
  `1601`, `1626`, `1630`, plus `_compute_layer_logits`).
  Each `astype` defaults to `copy=True` and was allocating a
  fresh F32 buffer per decoder block even though the weights
  were already F32. On Qwen3-14B Q4_K_M the gate_w astype
  alone measured ~107 ms per call (370 MB memcpy).
  Synthetic benchmark on the 14B projection path: per-layer
  projection work dropped 270 ms -> 65 ms (4.16x).
  End-to-end on Qwen3-0.6B Q4_K_M: top-4 projection time
  10.77 s -> 7.53 s (a ~30 % reduction).
- **`_SlidingDequantCache` + `enter_layer(idx)` eviction**
  for the dequant cache. Pre/post-layer tensors (embedding,
  norm, lm_head) are auto-immune; layer-indexed weights
  (parsed from `model.layers.N.` or
  `language_model.model.layers.N.` keys) evict as soon as the
  forwarder enters `N + stride` layers past them.
- **`_finish` `copy=False`** in `dequant/gguf.py`. The shared
  tail of every GGUF dequant function was paying a redundant
  output-buffer memcpy on the production F32 path.

### Removed

- **LayerAnalyzer** (the previous hidden-state-activity score).
  It produced misleading results because the final layer
  always scored high just from the projection, and the
  suggested subsets contradicted the early-exit suggestion.
  Replaced by `PredictionAnalyzer` which measures the model's
  actual next-token decision rather than the residual-stream
  movement.

### Performance findings

The detailed profiler confirmed the suspicion that the largest
specific cost in the forward pass is the **QKV / MLP concat +
astype** pattern: per layer, the forwarder allocates ~360 MB
(qkv) and ~740 MB (gateup) via `np.concatenate` + `.astype(
np.float32)`, even when the matmul itself is the same size.
The second-largest cost is the **autoregressive decode loop**
which re-encodes the entire prompt per step. See
`docs/performance-review.md` for the full analysis.

## [0.1.0] - 2026-07-31

First public release.

### Added

- **Streaming runtime** (`flatrun.runtime`): mmap-based weight
  access, layer scheduler with LRU eviction, per-layer KV cache,
  configurable memory manager.
- **Storage backends**:
  - `SafeTensorBackend` - hand-written parser, no third-party
    dependencies.
  - `GGUFBackend` - parses GGUF v3 in pure Python, including
    on-disk dequant for Q1_0, Q4_0, Q5_0, Q5_1, Q4_K, Q5_K,
    Q6_K, and Q8_0.
  - `MultiBackend` - composite backend for sharded checkpoints.
- **Reference forwarder** (`flatrun.model.qwen2`): pure NumPy
  Qwen2 / Llama / Qwen3 decoder with bfloat16, float16, and
  float32 weight dtypes, per-head Q/K RMSNorm for Qwen3, NEOX
  and NORM RoPE layouts picked automatically from the GGUF
  architecture, tied or untied LM head, grouped-query attention.
- **BPE tokenizer** (`flatrun.tokenizer`): byte-level BPE that
  accepts a HuggingFace `tokenizer.json` (with `vocab.json` +
  `merges.txt` fallback) and reads the embedded vocab + merges
  from a GGUF's `tokenizer.ggml.*` block.
- **CLI** (`python -m flatrun.cli`): auto-detects GGUF /
  SafeTensors / MLX-4bit from the path, applies the model's
  chat template, runs greedy or sampled generation, prints
  top-N candidates and timing.
- **Parity harness** (`tools/compare_to_lmstudio.py`): runs
  FlatRun in greedy mode and compares against an OpenAI-
  compatible endpoint.

### Verified

| Model | Quant | LM Studio match |
|---|---|---|
| Qwen2.5-Coder-0.5B | Q8_0 | byte-for-byte |
| SmolLM2-360M-Instruct | Q8_0 | byte-for-byte |
| Qwen3-0.6B | Q4_K_M | byte-for-byte |
| Bonsai-1.7B (PrismML 1-bit) | Q1_0 | koheren |
| SmolLM2-360M | Q4_K_M (requant) | koheren |
| SmolLM2-360M | Q6_K (requant) | koheren |
| SmolLM2-135M-Instruct | SafeTensors bf16 | koheren |
| Qwen2.5-0.5B | SafeTensors bf16 | koheren |

### Not yet supported (0.1.0)

- Qwen3.5 (hybrid linear + full attention / DeltaNet)
- Gemma 3 (per-MLP RMSNorm, gated attention, qk-norm gain)
- Phi-3, Gemma 2, Mistral forwarders
- CUDA / Metal / MLX kernels
- Incremental decode in the CLI (full prefill per step)
- SentencePiece tokenizer (LLaMA-1)
