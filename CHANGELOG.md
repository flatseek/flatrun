# Changelog

All notable changes to FlatRun are documented in this file. The
format is loosely [Keep a Changelog](https://keepachangelog.com/);
versions follow [Semantic Versioning](https://semver.org/).

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
    on-disk dequant for Q1_0, Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K,
    and Q8_0.
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

### Not yet supported

- Qwen3.5 (hybrid linear + full attention / DeltaNet)
- Gemma 3 (per-MLP RMSNorm, gated attention, qk-norm gain)
- Phi-3, Gemma 2, Mistral forwarders
- CUDA / Metal / MLX kernels
- Incremental decode in the CLI (full prefill per step)
- SentencePiece tokenizer (LLaMA-1)
