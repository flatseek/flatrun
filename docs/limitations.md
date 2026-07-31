# Limitations

FlatRun 0.1.0 is intentionally a thin runtime over a single
reference forwarder. The following are out of scope or only
partially supported.

## Architectures

The reference forwarder covers the Qwen2 / Llama family plus
Qwen3 (per-head Q/K RMSNorm) and Bonsai's Q1_0 1-bit format.
It does not (yet) implement:

- **Gemma 3** - per-MLP RMSNorm, gated attention,
  `query_pre_attn_scalar`, and the qk-norm gain `(1 + w)`.
  These need their own decoder block.
- **Qwen3.5 / Qwen3.5-MoE** - hybrid linear + full attention
  (DeltaNet), per-layer type selection, `attn_output_gate`.
  Requires a new forwarder class.
- **Phi-3**, **Gemma 2**, **Mistral** - the
  `flatrun.model.qwen2._decoder_block` would need a
  separate code path for each.

See [`model-matrix.md`](model-matrix.md) for the live list
of supported and unsupported models.

## Performance

- The forwarder is pure NumPy. A 0.5 B Q8_0 model generates
  around 25 tok/s on a single thread of a modern laptop CPU.
- There is no BLAS, no Metal kernel, no MLX decoder, no
  CUDA. Replacing the per-layer matmuls with a real kernel
  is the obvious optimisation path; the streaming layer
  itself is not on the critical path.
- The CLI re-runs the full prefill on every step instead of
  appending a single token to the KV cache. For long
  generations, an incremental `step_incremental` method
  would help.

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
  doesn't speak OpenTelemetry, Prometheus, etc.
