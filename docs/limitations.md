# Limitations

FlatRun is intentionally a thin runtime over a single
reference forwarder. The following are out of scope or only
partially supported.

## Architectures

The reference forwarder in `flatrun.model.qwen2` covers
Qwen2 / Qwen2.5 / Qwen3 / Qwen3.5 (full attention only) /
Llama / Gemma 3 / SmolLM2 / Bonsai (Q1_0). Each architecture
shares the same decoder block skeleton with small variations
(Gemma 3's `1 + weight` RMSNorm gain, Qwen3's per-head Q/K
norm, Qwen3.5's output gate). It does not (yet) implement:

- **Qwen3.5 / Qwen3.5-MoE linear (DeltaNet) layers** - the
  linear-attention path needs a recurrent state update with
  chunked delta-rule. The full-attention path works. The
  forwarder raises a clear `NotImplementedError` when it
  hits a `linear_attention` layer type so the failure mode
  is obvious.
- **Phi-3**, **Gemma 2**, **Mistral** - the
  `flatrun.model.qwen2._decoder_block` would need a
  separate code path for each.

See [`model-matrix.md`](model-matrix.md) for the live list
of supported and unsupported models.

## Performance

- The forwarder is pure NumPy. A 0.5 B Q8_0 model generates
  around 25 tok/s on a single thread of a modern laptop CPU.
  A Qwen3-14B Q4_K_M streams at ~1.5 tok/s on the same
  hardware.
- There is no BLAS, no Metal kernel, no MLX decoder, no
  CUDA. Replacing the per-layer matmuls with a real kernel
  is the obvious optimisation path; the streaming layer
  itself is not on the critical path.
- Per-layer QKV / MLP concat + astype allocates ~1 GB per
  forward pass on a 14B model. The memory manager keeps
  this bounded via `madvise(MADV_DONTNEED)` but the alloc
  itself is wasted work. See
  [`performance-review.md`](performance-review.md) for the
  full breakdown.
- The CLI re-runs the full prefill on every step instead of
  appending a single token to the KV cache. For long
  generations, an incremental `step_incremental` method
  would help. The detailed profiler makes this visible
  (`--profile-detailed`).
- Use `--profile-detailed` to see the per-operation breakdown
  of a forward pass. The summary aggregates by category
  (Attention, MLP, Tensor Loading, Dequantization, Norm,
  Sampling, Other) so the percentages tell a story.

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

- Single-threaded. FlatRun does not parallelise across
  cores; the matmuls are already large enough to fill one
  core.
- No autograd. FlatRun is inference only.
- No LoRA / adapter support. The forwarder reads the base
  weights and applies them directly.
- No model sharding strategy beyond what `MultiBackend`
  does at load time. The runtime doesn't split a single
  layer across devices.
- No metrics export. The CLI prints timing to stderr but
  doesn't speak OpenTelemetry, Prometheus, etc. Use
  `--profile-save PATH` to dump the profiler result as
  JSON and pipe to your own collector.
