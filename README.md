# FlatRun

**Streaming inference runtime for LLMs that don't fit in RAM.**

FlatRun treats model weights the same way an operating system treats
virtual memory: only the layer currently being executed is resident in
RAM, the rest stays on disk and is memory-mapped on demand. The goal
is *constant peak RAM regardless of model size*, not maximum throughput.

```text
Load current layer     Run inference     Release current layer
         ↓                                  ↓
      Inference                        Load next layer
                                            ↓
                                        Inference
```

FlatRun is **not** a new model format. It consumes existing
[SafeTensors](https://huggingface.co/docs/safetensors) and
[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
checkpoints and ships a streaming executor that you can drive with
``python -m flatrun.cli``.

---

## Why FlatRun exists

Most open-weight LLMs are now larger than the RAM of typical developer
machines. Existing runtimes expect the whole checkpoint to fit; when
it doesn't, you need a beefier machine or a more aggressive quantisation.

FlatRun is the third option: **stream the model from disk**. The
runtime only holds a single decoder layer (plus whatever the user
chooses to cache) at any moment, so a 70 B Q4 model runs comfortably
on a 16 GB laptop - at the cost of slower inference.

---

## What's in 0.1.0

- **Streaming runtime** (`flatrun.runtime`): mmap-based weight access,
  layer scheduler with LRU eviction, per-layer KV cache, custom
  memory manager with byte cap.
- **Storage backends** (`flatrun.backend`):
  - `SafeTensorBackend` - hand-written parser, no third-party deps.
  - `GGUFBackend` - parses GGUF v3 in pure Python, including
    on-disk dequantisation for the common block types.
  - `MultiBackend` - composite backend for sharded checkpoints.
- **Reference forwarder** (`flatrun.model.qwen2`): pure NumPy
  Qwen2 / Llama / Qwen3 decoder with the architectural features
  that matter for accuracy:
  - per-head RMSNorm on Q and K (Qwen3)
  - half-split (NEOX) and consecutive-pair (NORM) RoPE, picked
    automatically from the GGUF architecture
  - tied or untied LM head
  - grouped-query attention
  - bfloat16, float16, and float32 weight dtypes
- **GGUF dequant** (`flatrun.dequant`): Q1_0, Q4_0, Q5_0, Q5_1,
  Q4_K, Q5_K, Q6_K, Q8_0. All K-quants are vectorised transcriptions
  of the matching `dequantize_row_q*` in
  [llama.cpp ggml-quants.c](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c).
- **BPE tokenizer** (`flatrun.tokenizer`): byte-level BPE that
  accepts a HuggingFace `tokenizer.json` (with `vocab.json` +
  `merges.txt` fallback) and reads the embedded vocab + merges from a
  GGUF's `tokenizer.ggml.*` block.
- **CLI** (`python -m flatrun.cli`): single entry point that
  auto-detects GGUF / SafeTensors / MLX-4bit, applies the model's
  chat template, and runs greedy or sampled generation.

---

## Supported architectures

FlatRun decodes the GGUF `general.architecture` and the HF
`config.json` `architectures` field. The table below summarises what
runs correctly today.

| Architecture | Format | Status | Notes |
|---|---|---|---|
| Llama 1/2/3 | SafeTensors | ✅ | dense + GQA, tied or untied head |
| Llama 1/2/3 | GGUF | ✅ | uses NORM RoPE, picked automatically |
| Qwen2 / Qwen2.5 | SafeTensors | ✅ | bf16 / fp16 / fp32 |
| Qwen2 / Qwen2.5 | GGUF | ✅ | uses NEOX RoPE, bf16 / fp16 / fp32 |
| Qwen2.5-Coder | both | ✅ | identical to Qwen2.5 |
| Qwen3 | both | ✅ | per-head `q_norm` / `k_norm` applied |
| SmolLM2 | both | ✅ | Llama-arch with GQA |
| **Bonsai (PrismML 1-bit)** | GGUF Q1_0 | ✅ | 1.125 bpw, custom ggml type id 41 |
| Qwen3.5 / Qwen3.5-MoE | MLX | ❌ | hybrid linear + full attention (DeltaNet) |
| Gemma 3 | MLX | ❌ | per-MLP RMSNorm, gated attention, qk-norm gain |
| Phi-3 / Gemma 2 / Mistral | — | ❌ | not yet wired |

The model matrix we test against lives in
[`docs/model-matrix.md`](docs/model-matrix.md).

---

## Installation

```bash
git clone https://github.com/flatseek/flatrun.git
cd flatrun
make install         # editable install
make install-dev     # + pytest, mypy, ruff
```

FlatRun requires **Python ≥ 3.10** and `numpy`. There is no other
runtime dependency. The CLI is registered as a console script
(`flatrun`) by `pip install -e .`.

---

## Quick start

The CLI is the fastest way to verify FlatRun against a model you
already have on disk. It auto-detects GGUF / SafeTensors / MLX-4bit
from the path you point at.

```bash
# 1. Greedy one-liner with a GGUF (no chat template)
PYTHONPATH=src python3 -m flatrun.cli \
    --model ~/.lmstudio/models/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/smollm2-360m-instruct-q8_0.gguf \
    --prompt "The capital of France is" \
    --no-chat-template --max-new 12 --no-sample

# 2. Same model, full chat template + sampling
PYTHONPATH=src python3 -m flatrun.cli \
    --model ~/.lmstudio/models/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/smollm2-360m-instruct-q8_0.gguf \
    --prompt "Write a haiku about the sea." \
    --max-new 24 --temperature 0.7 --sample-top-k 40 --sample-top-p 0.9

# 3. SafeTensors HuggingFace checkpoint
PYTHONPATH=src python3 -m flatrun.cli \
    --model ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/*/ \
    --prompt "What is the capital of France?" \
    --max-new 10 --no-sample --top-k 5

# 4. JSON multi-turn
PYTHONPATH=src python3 -m flatrun.cli \
    --model <model_dir> \
    --messages-json '[{"role":"system","content":"You are a pirate."},{"role":"user","content":"Hello!"}]' \
    --max-new 16 --no-sample
```

### Programmatic use

```python
from flatrun import load_huggingface, KVCache, StreamingExecutor
from flatrun.model.qwen2 import Qwen2Config, make_qwen2_forwarder

loaded = load_huggingface("/path/to/model-dir")
cfg = Qwen2Config.from_hf_config(loaded.config.raw)
fwd = make_qwen2_forwarder(cfg)
sch = loaded.runtime.build_scheduler(
    loaded.manifest.layers,
    pre_layer_names=loaded.manifest.pre_layer,
    post_layer_names=loaded.manifest.post_layer,
)
ex = StreamingExecutor(sch, fwd, kv_cache=KVCache(capacity=4096))
result = ex.step(tokens=[1, 2, 3, 4])
logits = result.last_hidden            # (seq, vocab)
```

The executor runs the full forward pass: embed → 28 decoder blocks
→ final RMSNorm → LM head, with the KV cache growing incrementally
as ``tokens`` are appended.

### Interactive chat

```bash
PYTHONPATH=src python3 -m flatrun.cli chat \
    --model ~/.lmstudio/models/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/smollm2-360m-instruct-q8_0.gguf \
    --max-new 48 --temperature 0.5

You: Tell me a haiku about the sea.
Assistant: Silent waves, they sleep,
  Their hearts in stillness lie.
  The ocean's vast and deep,
  ...
You: exit
Bye.
```

Each turn is rendered through the model's chat template with the
full prior history included. Replies stop on the natural end-of-turn
marker (``<|im_end|>`` for Qwen, ``<|endoftext|>`` for GPT-style,
``</s>`` for LLaMA) or after ``--max-new`` tokens, whichever comes
first. Use ``--no-history`` to make every turn a one-shot call with
no prior context.

---

## CLI reference

The CLI exposes two subcommands:

```text
flatrun run   [shared-options] [--prompt TEXT | --messages-json JSON]
flatrun chat  [shared-options] [--no-history]
flatrun --help            # lists shared options + subcommands
flatrun run --help        # full help for the one-shot mode
flatrun chat --help       # full help for the REPL mode
```

Shared options (identical for ``run`` and ``chat``):

```text
Model input:
  --model PATH             path to a GGUF file, a SafeTensors directory,
                           or an MLX-4bit directory
  --tokenizer PATH         override the tokenizer directory
  --system TEXT            prepend a system turn in chat templates
  --no-chat-template       treat prompts as raw text, skip template

Runtime:
  --cache-mb N             memory cache cap in MiB (default 256)
  --quant NAME             override GGUF quant type

Generation:
  --max-new N              tokens to generate after the prompt
                           (run: default 1, chat: default 128)
  --temperature F          sampling temperature (default 0.11)
  --sample-top-k N         top-k filter at sample time (default 20)
  --sample-top-p F         nucleus filter (default 0.59)
  --min-p F                min-p filter (default 0.05)
  --repeat-penalty F       repetition penalty (default 1.1)
  --no-sample              greedy argmax, skip all sampling
  --seed N                 RNG seed (default time-seeded)
  --top-k N                print top-N next tokens after the run

Diagnostics:
  --profile                print per-step timing breakdown
```

`run`-only options: ``--prompt TEXT``, ``--messages-json JSON``.
`chat`-only options: ``--no-history``.

For backwards compatibility, calling ``flatrun --model ... --prompt ...``
without an explicit subcommand is treated as ``flatrun run``.

---

## Verifying correctness against llama.cpp

A parity harness ships in `tools/`. It runs FlatRun in greedy mode
and asks an OpenAI-compatible endpoint (LM Studio, llama-server)
for the same continuation; if the two diverge on the first token
you almost always have a weight, RoPE, or orientation bug.

```bash
python3 tools/compare_to_lmstudio.py \
    ~/.lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf \
    --server-model qwen2.5-coder-0.5b \
    --prompt "The capital of France is" -n 12
```

Successful runs on the matrix in
[`docs/model-matrix.md`](docs/model-matrix.md) all show
`agreed: true` with byte-for-byte text equality.

---

## How it works

```text
┌────────────────────────────────────────────────────┐
│  StreamingExecutor.step(tokens)                    │
│                                                    │
│  for each layer in manifest:                       │
│      scheduler.acquire(layer) -> {handle, ...}     │
│      forwarder(layer, handles, kv_cache) -> hidden │
│      scheduler.release(layer)                      │
│                                                    │
│  last layer's forwarder applies norm + LM head     │
│  and returns logits of shape (seq, vocab).         │
└────────────────────────────────────────────────────┘
```

* `scheduler` mmaps the per-layer handles, evicts the previous
  layer's mappings when the cache cap is exceeded, and lets the
  user-provided prefetch hook warm the next layer in the background.
* `forwarder` is a pure NumPy Qwen2 / Llama / Qwen3 forward pass
  that runs in float32 regardless of weight dtype. Numerical
  precision tracks llama.cpp to ~5 decimal places on Q8_0; the
  remaining gap on lower quants is the K-quant matmul
  approximations llama.cpp itself applies.
* `kv_cache` is layer-indexed and grows in O(1) per step. The
  forwarder uses absolute RoPE positions so the cache survives
  multi-step generation.

Full architecture notes are in
[`docs/architecture.md`](docs/architecture.md). The GGUF and
SafeTensors backend contracts are documented in
[`docs/backend.md`](docs/backend.md).

---

## Performance

FlatRun trades throughput for RAM. As a rule of thumb on a modern
laptop:

| Model | Backend | RAM peak | Tok/s (single thread) |
|---|---|---|---|
| Qwen2.5-Coder-0.5B Q8_0 | GGUF | ~640 MiB | ~25 |
| SmolLM2-360M Q8_0 | GGUF | ~400 MiB | ~40 |
| Qwen3-0.6B Q4_K_M | GGUF | ~512 MiB | ~12 |
| Bonsai-1.7B Q1_0 | GGUF | ~3 GiB | ~5 |

Pure-NumPy GEMM is the bottleneck; replacing `_apply_rope`,
`_decoder_block`, and the LM head with a BLAS or MLX kernel
typically doubles throughput on Apple Silicon. The streaming layer
itself (mmap, scheduler, KV cache) is not on the critical path.

---

## Limitations

See [`docs/limitations.md`](docs/limitations.md). The short version:

- CPU-only. No CUDA, no Metal kernel, no MLX decode.
- Pure NumPy forward - slow vs cuBLAS. The Qwen2 forwarder is
  intentionally simple so the streaming plumbing is easy to read.
- BPE tokenizer only - SentencePiece (LLaMA-1) is not implemented.
- Some chat templates fall back to a generic Qwen2 default when
  the model ships an exotic one.
- The CLI re-encodes the whole prompt each step (full prefill),
  not incremental decode.

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).

---

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md). 0.1.0 is the first public
release.
