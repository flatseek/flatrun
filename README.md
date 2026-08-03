<div align="center">

<img src="logo.svg" alt="Flatrun" width="64" height="64">

# Flatrun

**The streaming inference runtime for open-weight LLMs.**

<p align="center">
  <em>
Run GGUF, SafeTensors, and MLX models layer-by-layer directly from storage,
enabling inference on models larger than available system RAM.
  </em>
</p>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://github.com/flatseek/flatrun/workflows/Test/badge.svg)](https://github.com/flatseek/flatrun/actions)
[![PyPI version](https://img.shields.io/pypi/v/flatrun.svg)](https://pypi.org/project/flatrun/)


**GitHub:** https://github.com/flatseek/flatrun
&nbsp;&middot;&nbsp;
**Organization:** https://github.com/flatseek

<br>

**Part of the Flatseek ecosystem**

[Flatseek](https://github.com/flatseek/flatseek) (Keyword Search) •
[Flatvec](https://github.com/flatseek/flatvec) (Vector Search) •
[Flatask](https://github.com/flatseek/flatask) (RAG Runtime) •
[Flatlens](https://github.com/flatseek/flatlens) (Data Visualization) •
[Flatbuild](https://github.com/flatseek/flatbuild) (LLM Training) •
[Flattune](https://github.com/flatseek/flattune) (LLM Fine-Tuning) •
[Flatrun](https://github.com/flatseek/flatrun) (LLM Inference)

</div>

---
## See It In Action

Download **Flatbot-micro-4M**, the official demonstration model trained entirely from scratch with Flatbuild:

```bash
wget https://huggingface.co/flatseek/flatbot-micro-4M/resolve/main/model.gguf -O flatbot-micro-4M.gguf
```

Start an interactive chat:

```bash
flatrun chat --model flatbot-micro-4M.gguf --temp 0.2
```

Example session:

```text
Detected format: gguf
Building tokenizer from GGUF metadata (flatbot-micro-4M.gguf) ...
Tokenizer vocab: 516
Loaded model in 0.01 s; layers=6

Chat mode (max_new=128/turn, history=True).
Type your message; Ctrl-D (EOF) or 'exit' to quit.

You: Who are you?

Assistant:
Sure — I'm Flatbot — a small conversational assistant trained from
scratch using Flatbuild.
```

Want to expose the same model over HTTP? `flatrun serve` provides OpenAI-
and Anthropic-compatible APIs on the same port:

```bash
pip install "flatrun[serve]"

flatrun serve \
    --model flatbot-micro-4M.gguf \
    --port 8080
```

Example using the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="flatbot-micro-4M",
    messages=[
        {"role": "user", "content": "Who are you?"}
    ],
)

print(response.choices[0].message.content)
```

See [`docs/serve.md`](docs/serve.md) for the complete API reference, supported endpoints, and streaming response formats.

---
# Overview

**A pure Python and NumPy runtime for AI inference research and optimization.**

Flatrun is an experimental inference runtime built to explore how large language models can execute **without loading an entire checkpoint into memory**.

Instead of treating a model as a single monolithic file, Flatrun streams weights layer-by-layer directly from disk. Only the layer currently being executed resides in memory while the remaining weights stay memory-mapped on storage.

This streaming architecture allows models significantly larger than available system RAM to run on commodity hardware, trading throughput for dramatically lower memory usage.

---
# Why Flatrun?

Flatrun is designed for developers and researchers who want to understand how LLM inference works.

It provides a transparent streaming runtime where every stage—from tensor loading to KV cache management and forward execution—can be inspected, profiled, modified, and optimized.

---
# Highlights

- Streaming layer-by-layer execution
- Constant-memory architecture
- Pure Python + NumPy runtime
- Optional native C++ acceleration
- GGUF, SafeTensors, and MLX support
- Q1/Q4/Q5/Q6/Q8 quantization
- Python API and interactive chat
- OpenAI- and Anthropic-compatible HTTP server (`flatrun serve`)
- Layer profiler and memory tracing
---

# Supported Models

| Family | GGUF | SafeTensors | MLX |
|---------|:---:|:-----------:|:---:|
| Llama 1 / 2 / 3 | ✅ | ✅ | — |
| Qwen2 | ✅ | ✅ | — |
| Qwen2.5 | ✅ | ✅ | — |
| Qwen2.5 Coder | ✅ | ✅ | — |
| Qwen3 | ✅ | ✅ | — |
| Qwen3.5 | ✅ | — | — |
| SmolLM2 | ✅ | ✅ | — |
| Gemma 3 | — | — | ✅ |
| Bonsai Q1_0 | ✅ | — | — |

Support for additional architectures is continuously expanding.

---

# Installation

Install from PyPI:

```bash
pip install flatrun
```
Or clone + editable install:


```bash
git clone https://github.com/flatseek/flatrun.git

cd flatrun

make install
```

Requirements:

- Python 3.10+
- NumPy

---

# Quick Start

Run a GGUF model:

```bash
flatrun \
    --model model.gguf \
    --prompt "The capital of France is" \
    --max-new 32
```

Interactive chat:

```bash
flatrun chat \
    --model model.gguf
```

Run a SafeTensors checkpoint:

```bash
flatrun \
    --model /path/to/model \
    --prompt "Explain quantum computing."
```

---

## Backend Selection

Flatrun provides two interchangeable execution backends.

### Python Backend (Default)

The reference runtime is implemented entirely in pure Python and NumPy, making every stage of inference easy to inspect, profile, and modify.

```bash
flatrun \
    --backend python \
    --model model.gguf \
    --prompt "Hello"
```

This is the default backend, so `--backend python` can be omitted.

### Native Backend

For higher performance, Flatrun can execute performance-critical operations using the native C++ backend located in `src/flatrun_native`.

```bash
flatrun \
    --backend native \
    --model model.gguf \
    --prompt "Hello"
```

The native backend preserves the same streaming execution pipeline while accelerating compute-intensive operations such as quantized GEMM and dequantization.

Backend options:

- `--backend python` *(default)* — Pure Python and NumPy runtime.
- `--backend native` — Native C++ accelerated runtime.

---

# Python API

Flatrun is fully scriptable. The same runtime that powers the CLI is
exposed as a Python library — pick a local model file, hand it a
prompt, and read the logits back. Nothing in the public surface
requires downloading a model from a hub.

## Full minimal example

```python
from flatrun import KVCache, StreamingExecutor, load_huggingface
from flatrun.model import make_qwen2_forwarder
from flatrun.model.sampling import Sampler
from flatrun.runtime.backend import get_backend
from flatrun.tokenizer import auto_load

# 1. Open a local GGUF
loaded = load_huggingface("/Users/me/models/qwen3-0.6b-q4_k_m.gguf")

# 2. Tokenize
tokenizer = auto_load("/Users/me/models/qwen3-0.6b-q4_k_m.gguf")
prompt = "The capital of France is"
tokens = list(tokenizer.encode(prompt))

# 3. Build the executor
backend = get_backend("python")
forwarder = make_qwen2_forwarder(
    loaded.config, enable_dequant_cache=True, backend=backend,
)
kv_cache = KVCache(capacity=4096)
scheduler = loaded.runtime.build_scheduler(loaded.manifest.layers)
executor = StreamingExecutor(scheduler, forwarder, kv_cache=kv_cache)

# 4. Prefill + decode
sampler = Sampler(temperature=0.0001, top_k=1)
step = executor.step(tokens)
next_id = sampler.sample(step.last_hidden[-1])
out = list(tokens) + [next_id]
for _ in range(32):
    step = executor.step([next_id])
    next_id = sampler.sample(step.last_hidden[-1])
    out.append(next_id)

# 5. Decode
print(tokenizer.decode(out))
```

---
# CLI

```bash
flatrun run
flatrun chat
```

Supports:

- Text generation and sampling
- Python and native backends
- Runtime and cache configuration
- Quantization overrides
- Layer selection
- Profiling and debugging

See `flatrun run --help` for the complete command reference.

---
# How It Works

```text
Model on Disk
      │
      ▼
 Load Layer
      │
      ▼
 Execute
      │
      ▼
 Update KV
      │
      ▼
Release Layer
      │
      ▼
 Next Layer
```

Instead of loading an entire checkpoint into memory, Flatrun streams one decoder layer at a time. Only the active layer and KV cache remain resident in memory, keeping peak RAM nearly constant regardless of model size.

---
# Performance

Flatrun prioritizes memory efficiency over maximum throughput.

Typical performance on modern Apple Silicon:

| Model | Peak RAM | Speed |
|--------|----------:|------:|
| SmolLM2-360M Q8 | ~400 MB | ~40 tok/s |
| Qwen3-0.6B Q4 | ~512 MB | ~12 tok/s |
| Qwen3-14B Q4 | ~2–30 GB | ~0.7–1.5 tok/s |

Performance depends on:

- Model size
- Quantization
- Storage bandwidth
- Cache configuration

Profile every stage of execution using:

```bash
flatrun \
    --profile-detailed \
    --profile-save profile.json
```

---

# Native Backend

While the reference runtime is intentionally written in pure Python, Flatrun also includes an optional native backend located in `src/flatrun_native`.

The native backend accelerates performance-critical components using C++ while preserving the exact same streaming architecture and runtime API.

Current focus includes:

- Quantized GEMM kernels (fused dequant + matmul)
- Per-quant coverage: **Q4_K**, **Q6_K**, **Q8_0**
- SIMD vectorization (ARM NEON 128-bit intrinsics)
- Multi-threaded row-parallel matmul (std::thread pool, one per call)
- Batched matmul API for prefill (no per-token Python overhead)
- Reduced Python overhead via pybind11 marshalling

The native backend is selected with `--backend native`. When the C++
extension is not built (no pybind11 toolchain available), or when a
specific kernel rejects an unusual layout, the call falls back to
the numpy path transparently — `--backend native` is always safe.

Each supported quant type has a parity test in
`src/flatrun_native/tests/`:

| Quant | Test | Tolerance |
|-------|------|----------:|
| Q4_K  | `test_q4_k_parity.py` | ~3e-6 max rel |
| Q6_K  | `test_q6_k_parity.py` | ~3e-4 max rel |
| Q8_0  | `test_q8_0_parity.py` | ~2e-5 max rel |

See [`docs/native.md`](docs/native.md) for the kernel design notes
and the multi-threading / batched-API rationale.

---

# Research Focus

Flatrun serves as a platform for exploring new ideas in AI runtime design.

Current research areas include:

- Streaming inference
- Layer scheduling
- Memory-aware execution
- Quantized inference
- Dequantization optimization
- KV cache architectures
- Native kernel acceleration
- Runtime instrumentation
- Pure NumPy performance

The codebase is intentionally designed to be readable, modular, and easy to modify, making it suitable for experimentation, benchmarking, and runtime research.

---

# Limitations

Current limitations include:

- CPU-only execution
- Pure NumPy reference runtime
- No CUDA backend
- No Metal kernels
- Lower throughput than GPU runtimes
- Some architectures are still under active development

Flatrun is built primarily for experimentation, research, and runtime innovation rather than large-scale production serving.

---

# License

Apache License 2.0

See `LICENSE`.
