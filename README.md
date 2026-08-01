# Flatrun

**Streaming inference runtime for LLMs that don't fit in RAM.**

Flatrun treats model weights the same way an operating system treats
virtual memory: only the layer currently being executed is resident
in RAM, the rest stays on disk and is memory-mapped on demand. The
goal is **constant peak RAM regardless of model size**, not
maximum throughput.

```text
Load current layer     Run inference     Release current layer
         ↓                                  ↓
      Inference                        Load next layer
                                            ↓
                                        Inference
```

Flatrun is **not** a new model format. It consumes existing
[SafeTensors](https://huggingface.co/docs/safetensors) and
[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
checkpoints and ships a streaming executor that you can drive with
the `flatrun` CLI.

---

## Table of contents

- [Why Flatrun exists](#why-flatrun-exists)
- [What this version ships](#what-this-version-ships)
- [Supported architectures](#supported-architectures)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [How it works](#how-it-works)
- [Performance story](#performance-story)
- [The optimisation journey](#the-optimisation-journey)
- [What's at the pure-NumPy ceiling](#whats-at-the-pure-numpy-ceiling)
- [Limitations](#limitations)
- [License](#license)
- [Changelog](#changelog)

---

## Why Flatrun exists

Most open-weight LLMs are now larger than the RAM of typical developer
machines. Existing runtimes expect the whole checkpoint to fit; when
it doesn't, you need a beefier machine or a more aggressive
quantisation.

Flatrun is the third option: **stream the model from disk**. The
runtime only holds a single decoder layer (plus whatever the user
chooses to cache) at any moment, so a 70 B Q4 model runs comfortably
on a 16 GB laptop — at the cost of slower inference.

---

## What this version ships

### Streaming runtime
- **mmap-based tensor access** (`flatrun.runtime.memory.MemoryManager`).
  Every tensor handle is a zero-copy view into the source file. On
  Linux/macOS, mmap'd tensors larger than 1 MiB are released back
  to the OS via `madvise(MADV_DONTNEED)` when the handle closes,
  so RSS stays bounded by the per-layer working set rather than the
  cumulative touched-mmap.
- **Layer scheduler** (`flatrun.runtime.scheduler.LayerScheduler`)
  with LRU eviction and a pre-fetch hook for callers who want
  to warm the next layer asynchronously.
- **Preallocated growing F32 KV cache**
  (`flatrun.runtime.kv_cache.KVCache`). Each layer owns a single
  `(capacity, kv_heads, head_dim)` F32 buffer pair; `append`
  writes the next slot, `stack` is a zero-copy view of the live
  region. Replaces the previous `np.stack`-per-call design that
  cost an `O(T)` allocation on every decoder step.
- **Dequant cache** (`flatrun.runtime.executor`,
  `--dequant-cache on` by default). The F32 decoder outputs are
  memoised by on-disk tensor name so the second-and-later decode
  steps don't redo the dequant work. Bound the cache to "current +
  next layer" with `--dequant-cache-stride N` on memory-constrained
  hosts (`--dequant-cache-stride 2` keeps ~2 layers' worth of F32
  data resident on a 14 B model where unbounded caching needs ~7 GB).

### Storage backends (`flatrun.backend`)
- `SafeTensorBackend` — hand-written SafeTensors parser, zero
  third-party deps.
- `GGUFBackend` — pure-Python GGUF v3 reader + tensor-name
  translation so the same manifest builder works for both.
- `MultiBackend` — composite backend for sharded checkpoints.

### Reference forwarder (`flatrun.model.qwen2`)
Pure NumPy decoder for the Qwen2 / Llama / Qwen3 / Qwen3.5 / Gemma 3
family with:

- per-head Q/K RMSNorm (Qwen3, Qwen3.5)
- half-split (NEOX) and consecutive-pair (NORM) RoPE, picked
  automatically from the GGUF architecture
- tied or untied LM head
- grouped-query attention
- bfloat16, float16, and float32 weight dtypes
- GEMM dispatches to Accelerate on macOS, the OpenBLAS-compatible
  BLAS elsewhere — verified via `np.show_config()`

### Dequant (`flatrun.dequant.gguf`, `flatrun.dequant.mlx`)
Pure NumPy implementations of:

- GGUF K-quants: `Q4_K`, `Q5_K`, `Q6_K`
- GGUF small-block: `Q4_0`, `Q5_0`, `Q5_1`, `Q8_0`
- GGUF 1-bit: `Q1_0` (Bonsai / PrismML)
- MLX 4-bit (`weight` + `scales` + `biases` triples, used by
  Gemma 3 checkpoints)

All K-quants are vectorised transcriptions of the matching
`dequantize_row_q*` in
[llama.cpp ggml-quants.c](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c).
Per-call output is F32 (zero-copy `astype(F32, copy=False)`) regardless
of the dtype the caller asked for; `_finish` lives at the tail of
every dequant function and avoids the redundant output-buffer memcpy
the early prototype paid.

### Diagnostics

- `--profile` — per-step timing breakdown.
- `--profile-detailed` — per-layer microsecond breakdown of every
  forward-pass operation; aggregates to category-level
  percentages at the end (Attention, MLP, Tensor Loading,
  Dequantization, Norm, Sampling, Residual, Other). JSON via
  `--profile-save PATH`.
- `--debug` — per-token debug table per layer (norm, delta,
  stable, influence, entropy, confidence, rank_by_norm,
  rank_delta, rank_stable) plus a Prediction Evolution summary
  that runs the final norm + LM head at every layer and tracks
  the next-token prediction. JSON via `--debug-save-analysis PATH`.
- `--memory-trace` — per-layer RSS / Python heap / KV / Dequant /
  hidden size.

### CLI

- `flatrun run` — single-shot generation. For backwards
  compatibility, `flatrun --model ... --prompt ...` without an
  explicit subcommand is treated as `flatrun run`.
- `flatrun chat` — REPL that accepts the same arguments plus a chat
  loop.

The CLI is registered as a console script (`flatrun`) by
`pip install -e .`. If you prefer not to install:

```bash
PYTHONPATH=src flatrun --help  # uses the source tree directly
```

---

## Supported architectures

Flatrun decodes the GGUF `general.architecture` and the HF
`config.json` `architectures` field.

| Architecture | Format | Status | Notes |
|---|---|---|---|
| Llama 1/2/3 | SafeTensors | ✅ | dense + GQA, tied or untied head |
| Llama 1/2/3 | GGUF | ✅ | uses NORM RoPE, picked automatically |
| Qwen2 / Qwen2.5 | SafeTensors | ✅ | bf16 / fp16 / fp32 |
| Qwen2 / Qwen2.5 | GGUF | ✅ | uses NEOX RoPE, bf16 / fp16 / fp32 |
| Qwen2.5-Coder | both | ✅ | identical to Qwen2.5 |
| Qwen3 | both | ✅ | per-head `q_norm` / `k_norm` applied |
| Qwen3.5 (full attention layers) | GGUF | ✅ | output gate supported when weight present |
| Qwen3.5 (linear / DeltaNet layers) | GGUF | ⚠️ | raises clear error; not implemented |
| SmolLM2 | both | ✅ | Llama-arch with GQA |
| Gemma 3 | MLX | ✅ | per-MLP RMSNorm, gated attention, qk-norm gain |
| **Bonsai (PrismML 1-bit)** | GGUF Q1_0 | ✅ | 1.125 bpw, custom ggml type id 41 |
| Phi-3 / Gemma 2 / Mistral | — | ❌ | not yet wired |

The model matrix we test against lives in
[`docs/model-matrix.md`](docs/model-matrix.md).

---

## Installation

```bash
git clone https://github.com/tenslab/flatrun.git
cd flatrun
make install         # editable install
make install-dev     # + pytest, mypy, ruff
```

Flatrun requires **Python ≥ 3.10** and `numpy`. There is no other
runtime dependency. The CLI is registered as a console script
(`flatrun`) by `pip install -e .`.

---

## Quick start

The CLI is the fastest way to verify Flatrun against a model you
already have on disk. It auto-detects GGUF / SafeTensors / MLX-4bit
from the path you point at.

```bash
# 1. Greedy one-liner with a GGUF (no chat template)
flatrun \
    --model ~/.lmstudio/models/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/smollm2-360m-instruct-q8_0.gguf \
    --prompt "The capital of France is" \
    --no-chat-template --max-new 12 --no-sample

# 2. Same model, full chat template + sampling
flatrun \
    --model ~/.lmstudio/models/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/smollm2-360m-instruct-q8_0.gguf \
    --prompt "Write a haiku about the sea." \
    --max-new 24 --temperature 0.7 --sample-top-k 40 --sample-top-p 0.9

# 3. SafeTensors HuggingFace checkpoint
flatrun \
    --model ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/*/ \
    --prompt "What is the capital of France?" \
    --max-new 10 --no-sample --top-k 5

# 4. JSON multi-turn
flatrun \
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

The executor runs the full forward pass: embed → N decoder
blocks → final RMSNorm → LM head, with the KV cache growing
incrementally as `tokens` are appended.

### Interactive chat

```bash
flatrun chat \
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
full prior history included. Replies stop on the natural
end-of-turn marker (`<|im_end|>` for Qwen, `<|endoftext|>` for
GPT-style, `</s>` for LLaMA) or after `--max-new` tokens,
whichever comes first. Use `--no-history` to make every turn a
one-shot call with no prior context.

---

## CLI reference

```text
flatrun --help            # lists shared options + subcommands
flatrun run --help        # full help for the one-shot mode
flatrun chat --help       # full help for the REPL mode
```

`flatrun run [shared-options] [--prompt TEXT | --messages-json JSON]`
— single-shot generation. Default subcommand when no subcommand is
given.

`flatrun chat [shared-options] [--no-history]`
— REPL that accepts the same arguments plus a chat loop.

Shared options:

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
  --dequant-cache [on|off]  keep dequantised F32 weights across steps
                           (default on). Trades RAM for speed: bounds
                           RAM with --dequant-cache-stride N below.
  --dequant-cache-stride N keep only the last N layers in the dequant
                           cache. Default is unbounded (every decoded
                           tensor held for the process lifetime).

Layer selection:
  --max-layers N           use only the first N decoder layers
  --layers LIST            custom subset of decoder layers, with
                           inclusive ranges (e.g. 0-6,19-24,34-39)
                           and comma separators (e.g. 0-6,8,11-12)

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
  --profile-detailed       per-layer microsecond breakdown of every
                           forward-pass operation; aggregates to
                           category-level percentages at the end
  --profile-save PATH      persist the detailed profiler result
                           (JSON); implies --profile-detailed
  --debug                  per-token debug table per layer +
                           Prediction Evolution summary
  --debug-include-special  show special tokens in the per-token table
  --debug-max-token-rows N max tokens shown per layer (default 16)
  --debug-save-analysis PATH  persist the Prediction Evolution
                              summary (JSON); implies --debug
  --memory-trace           per-layer RSS / Python heap / KV / Dequant
                           / hidden size
```

`run`-only options: `--prompt TEXT`, `--messages-json JSON`.
`chat`-only options: `--no-history`.

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

- **Scheduler** mmaps the per-layer handles, evicts the
  previous layer's mappings when the cache cap is exceeded,
  and lets the user-provided pre-fetch hook warm the next layer
  in the background.
- **Forwarder** is a pure NumPy Qwen2 / Llama / Qwen3 forward
  pass that runs in float32 regardless of weight dtype.
  Numerical precision tracks llama.cpp to ~5 decimal places on
  Q8_0; the remaining gap on lower quants is the K-quant matmul
  approximation llama.cpp itself applies.
- **KV cache** uses one (T, h, d) F32 buffer pair per layer;
  `append` writes the next slot, `stack` is a zero-copy view.
- **Dequant cache** is keyed by on-disk tensor name. With
  `--dequant-cache on` (default) and the sliding window
  unbound, every decoded F32 weight is held for the process
  lifetime; with `--dequant-cache-stride N`, only the last N
  layers stay resident.

Full architecture notes are in
[`docs/architecture.md`](docs/architecture.md). The GGUF and
SafeTensors backend contracts are documented in
[`docs/backend.md`](docs/backend.md). Layer streaming details
(including the dequant cache layout) are in
[`docs/streaming.md`](docs/streaming.md).

---

## Performance story

Pure-NumPy GEMM dispatches to Apple Accelerate on macOS and the
OpenBLAS-compatible BLAS on Linux. Numerical tracking on a recent
Apple M-class laptop:

| Model | Backend | RAM peak | Tok/s (single thread) | Notes |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B Q8_0 | GGUF | ~640 MiB | ~25 | |
| SmolLM2-360M Q8_0 | GGUF | ~400 MiB | ~40 | |
| Qwen3-0.6B Q4_K_M | GGUF | ~512 MiB | ~12 | `--dequant-cache on` |
| Qwen3-14B Q4_K_M | GGUF | ~30 GiB | ~1.5 | `--dequant-cache on` |
| Qwen3-14B Q4_K_M | GGUF (memory-bound) | ~2.0 GiB | ~0.7 | `--dequant-cache-stride 2` |
| Bonsai-1.7B Q1_0 | GGUF | ~3 GiB | ~5 | 1-bit, custom ggml type 41 |

Numbers above are steady-state on `--no-sample --max-new 12` with
one prompt; the throughput numbers include the prefill cost.

The detailed profiler is the source of truth for any new
bottlenecks:

```bash
flatrun run --profile-detailed --profile-save profile.json \
    --prompt "halo" --model /path/to/14B --max-new 1
```

The full review of the forward pass is in
[`docs/performance-review.md`](docs/performance-review.md). The
short version: pure-NumPy GEMM is already at the ceiling for
the projection + attention paths; remaining speedup needs a
compiled kernel (Numba / C / Metal) — see the next section.

---

## The optimisation journey

The codebase went through several audit cycles before this
release. Each audit identifies *one* bottleneck, fixes it,
and re-measures. The pattern is intentional — every
non-obvious line in the forwarder exists because the
profiler caught something specific, not because of an
anticipated bottleneck.

The audit trail (each item below is one cycle):

1. **`madvise(MADV_DONTNEED)` on tensor close** for tensors ≥ 1
   MiB. Without this the page cache held the touched mmap
   pages forever and RSS grew past 30 GB on a 14 B model
   even with the cache cap honoured.
2. **`--dequant-cache on` as the default**. The previous
   default (`off`) re-dequantised every layer on every
   decode step. With cache-on the dequantisation share
   dropped to single-digit percentages after the first step.
3. **Sliding-window bound on the dequant cache** (`_SlidingDequantCache`).
   The unbounded cache needs ~7 GB of F32 heap on a 14 B
   Q4_K_M model. The sliding cache evicts layer-indexed
   tensors when the forwarder enters `N + stride` layers
   past them; pre/post-layer tensors (embedding, norm,
   lm_head) are auto-immune.
4. **Drop redundant `.astype(np.float32)` in the projection
   matmul hot path.** The early prototype wrapped every
   weight in `.astype(np.float32)` before the matmul;
   `astype` defaults to `copy=True`, so even when the
   weight was already F32 it allocated a fresh buffer
   (measured 107 ms per `gate_w.astype(F32)` call on Qwen3-14B
   Q4_K_M). Replacing with `.astype(np_dtype, copy=False)` at
   14 hot-path sites gave **4.16× speedup** in a synthetic
   benchmark on 14 B shapes; end-to-end on Qwen3-0.6B
   Q4_K_M the top-4 projection time fell ~30 %.
5. **Drop redundant `.astype(np.float32)` in `_finish`** — the
   shared tail of every GGUF dequant function. Single
   one-line change; on Qwen3-14B Q4_K_M dequant this saves
   ~30 ms per call (~365 MB of F32 memcpy avoided).
6. **Fuse SiLU(gate)·up into a single F32 buffer** via an
   in-place `np.exp` → `+ 1.0` → `np.reciprocal` →
   `np.multiply(_, gate)` → `np.multiply(_, up)` chain. The
   previous code issued five separate NumPy expressions and
   allocated five intermediate F32 buffers. **1.40-1.56×
   speedup** on the post-gateup stage.
7. **Fix float64 leak in attention scale multiplication.**
   The production code computed `scale = 1.0 / np.sqrt(...)`
   which is a Python float (float64); multiplying an F32
   attention tensor by it promoted the result to F64 and
   silently leaked F64 through the rest of the attention
   path (mask add, softmax, av matmul, residual,
   *MLP downstream*). Cast `scale` to `np.float32` at the
   top of each decoder block.
8. **Rewrite the KV cache as a per-layer preallocated growing
   F32 buffer** (`_LayerKV`). The previous implementation
   stored past K/V as a Python list of per-token `(h, d)`
   arrays and rebuilt the full history via `np.stack` on
   every `stack(layer)` call. The new design folds one
   (T, h, d) buffer pair per layer; `append` writes the
   next slot, `stack` is a view. Measured isolated speedup
   on `stack()` alone: **61× (past=16) to 12,700×
   (past=4096)**. End-to-end attention share fell from
   ~36 % to ~27 % on Qwen3-0.6B Q4_K_M.

Each cycle's audit script lives in `/tmp/` (one per topic:
`projection_audit.py`, `attention_audit.py`,
`dequant_audit.py`). They use `time.perf_counter` and
`tracemalloc` to give microsecond-precision per-stage
breakdowns of the production path and dump intermediate
tensor layouts (dtype / shape / strides / contiguous /
OWNDATA).

---

## What's at the pure-NumPy ceiling

The audit cycle sized each part of the forward pass.
Headline:

| Path | BLAS / dispatch share | At the ceiling? |
|------|----------------------:|-----------------|
| Projection matmul (qkv/o/gateup/down) | 85-97 % | **Yes** — minimal Python dispatch overhead remains |
| Attention matmul (qk + av) | 70-85 % | Near — RoPE, RMSNorm, softmax each have non-BLAS share |
| KV cache stack | ~12,700× over old `np.stack` | **Yes** — zero-copy view |
| SiLU chain | 1.40-1.56× over previous 5-allocation version | **Yes** — single F32 buffer |
| Dequant (Q4_K / Q5_K / Q6_K) | dispatch-bound at ~0.9-1.2 GB/s | **Not at ceiling** — needs a compiled kernel |
| Projection matmul shape churn (`np.concatenate` for qkv_w, gateup_w) | — | **Not at ceiling** — caching the concat is a clean win |
| GQA repeat (`np.repeat(k_hist, head_group, axis=1)`) | — | **Not at ceiling** — strided broadcast view is plausible |

### Realistic remaining ceiling without compiled kernels

If every **Low- or Medium-complexity pure-NumPy** item
below ships:

| # | Optimisation | Est. additional total-runtime speedup | Complexity |
|--:|---|---:|---|
| 1 | Cache `qkv_w` and `gateup_w` `np.concatenate`s across decoder steps (extension of Task #12's sliding cache) | ~8-12 % | Low |
| 2 | Replace `gqa_repeat` `np.repeat` with strided broadcast view (validates against Accelerate sgemm dispatch) | ~5-10 % | Medium |
| 3 | Cache `q/k` `RMSNorm` weights per layer (already cached, but the `as_numpy().astype(F32, copy=False)` lookup could reuse the cached F32 from `dequant_cache`) | < 1 % | Low |
| 4 | FP16 weight storage (halve cache bytes and matmul reads) | 1.5-2× on memory-bound decoder paths | Medium |
| **Total achievable without compiled kernels** | | **~15-20 %** | |

### Beyond pure NumPy

A 3-10× structural speedup requires a compiled kernel:

- **Numba JIT** (easiest fit; bypasses the Python ↔ NumPy dispatch
  boundary).
- **Cython + typed memoryview**.
- **Quantized matmul** (skip dequant entirely; do `int4 @ fp16`
  directly via a custom kernel). Requires a compiled operator.
- **MPS / Metal Performance Shaders** on Apple Silicon (for
  FP16 matmul and fused quantized kernels).

None of these are in scope for this release. The pure-NumPy
path is the explicit baseline; the docs flag where the
bottleneck moves once the kernels land.

---

## Limitations

See [`docs/limitations.md`](docs/limitations.md). The short
version:

- CPU-only. No CUDA, no Metal kernel, no MLX decode.
- Pure NumPy forward — slow vs cuBLAS. A 0.5 B Q8_0 model
  generates around 25 tok/s on a single thread of a modern
  laptop CPU. A Qwen3-14B Q4_K_M streams at ~1.5 tok/s on
  the same hardware.
- BPE tokenizer only — SentencePiece (LLaMA-1) and
  WordPiece (BERT) are not implemented.
- Some chat templates fall back to a generic Qwen2 default
  when the model ships an exotic one.
- The CLI re-encodes the whole prompt each step (full
  prefill), not incremental decode.
- Some larger models (~7 B+) currently produce unstable
  activations or incorrect logits on hidden states past
  the first ~6 layers. Use `--debug` to see the per-layer
  mean / std / L2 norm / position cosine / adjacent-row
  diff / NaN-Inf count and pin down where the drift
  starts; file an issue with that layer number and the
  model hash so a maintainer can fix it.

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).

---

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md). The codebase is at
`0.1.0`; there has not yet been a public release.
