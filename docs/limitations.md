# Limitations

Flatrun is intentionally a thin runtime over a single
reference forwarder. The following are out of scope or only
partially supported.

## Architectures

The reference forwarder in `flatrun.model.qwen2` covers
Qwen2 / Qwen2.5 / Qwen3 / Qwen3.5 (full attention only) /
Llama / Gemma 3 / SmolLM2 / Bonsai (Q1_0). Each architecture
shares the same decoder block skeleton with small variations
(Gemma 3's `1 + weight` RMSNorm gain, Qwen3's per-head Q/K
norm, Qwen3.5's output gate). It does not (yet) implement:

- **Qwen3.5 / Qwen3.5-MoE linear (DeltaNet) layers** — the
  linear-attention path needs a recurrent state update with
  chunked delta-rule. The full-attention path works. The
  forwarder raises a clear `NotImplementedError` when it
  hits a `linear_attention` layer type so the failure mode
  is obvious.
- **Phi-3**, **Gemma 2**, **Mistral** — the
  `flatrun.model.qwen2._decoder_block` would need a
  separate code path for each.

See [`docs/model-matrix.md`](model-matrix.md) for the live
list of supported and unsupported models.

## Performance

- The forwarder is pure NumPy dispatching to Accelerate on
  macOS (or the OpenBLAS-compatible BLAS on Linux). A 0.5 B
  Q8_0 model generates around 25 tok/s on a single thread of
  a modern laptop CPU. A Qwen3-14B Q4_K_M streams at ~1.5
  tok/s on the same hardware.
- There is no Metal kernel, no MLX decoder, no CUDA.
  Replacing the per-layer matmuls with a real kernel is
  the obvious optimisation path; the streaming layer
  itself is not on the critical path. The projection path
  is currently 85-97 % BLAS time on the per-stage audit;
  the dequant kernel itself is **Python-dispatch bound**
  (~0.9-1.2 GB/s effective bandwidth on Q4_K, an order
  of magnitude below the 30-40 GB/s M-class ceiling) — only
  a compiled kernel (Numba / C / Metal) will unlock a
  structural speedup there.
- The remaining pure-NumPy items at this point are the
  `np.concatenate` for `qkv_w` / `gateup_w` (caching them
  across decoder steps would save ~8-12 % total runtime)
  and `gqa_repeat`'s `np.repeat` (replacing with a strided
  broadcast view would save ~5-10 % attention share).
  See [`docs/performance-review.md`](performance-review.md)
  for the full audit chain.
- The CLI re-runs the full prefill on every step instead of
  appending a single token to the KV cache. For long
  generations, an incremental `step_incremental` method
  would help. The detailed profiler makes this visible
  (`--profile-detailed`).
- Use `--profile-detailed` to see the per-operation breakdown
  of a forward pass. The summary aggregates by category
  (Attention, MLP, Tensor Loading, Dequantization, Norm,
  Sampling, Residual, Other) so the percentages tell a
  story.

## Tokenization

- BPE only. SentencePiece (LLaMA-1) and WordPiece (BERT)
  are not implemented.
- The chat template is parsed with a minimal Jinja subset;
  exotic templates (function-calling, multi-system) fall
  back to a generic Qwen2 ChatML layout.
- Streaming decode replaces partial UTF-8 multibyte
  sequences with U+FFFD in the printed output. The token
  IDs are unaffected; LM Studio also emits replacement
  characters in the same situation.

## Sampling

- No beam search, no speculative decoding, no KV-cache
  sharing across requests.
- The reference `Sampler` in `flatrun.model.sampling` is a
  straightforward temperature + top-k + top-p + min-p +
  repetition-penalty pipeline. If you want something
  fancier (DRY, XTC, mirostat), bring your own.

## Operational

- Single-threaded. Flatrun does not parallelise across
  cores; the matmuls are already large enough to fill one
  core (Accelerate is multi-threaded internally).
- No autograd. Flatrun is inference only.
- No LoRA / adapter support. The forwarder reads the base
  weights and applies them directly.
- No model sharding strategy beyond what `MultiBackend`
  does at load time. The runtime doesn't split a single
  layer across devices.
- No metrics export. The CLI prints timing to stderr but
  doesn't speak OpenTelemetry, Prometheus, etc. Use
  `--profile-save PATH` to dump the profiler result as
  JSON and pipe to your own collector.
- The unbounded dequant cache (`--dequant-cache on`
  without `--dequant-cache-stride`) on 14 B+ Q4_K_M
  models needs ~7 GB of F32 heap. If the host RAM is
  constrained, pass `--dequant-cache off` (pure
  streaming, slower) or `--dequant-cache-stride 2`
  (current + next layer only, with cold-cache misses at
  the start of every decode step).
