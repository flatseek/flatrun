# Architecture

FlatRun is a thin runtime that sits between a storage backend
(either SafeTensors or GGUF) and a user-supplied forward pass.
Three principles drive every design decision:

1. **Storage backends are pluggable.** The runtime never parses
   SafeTensors, GGUF, or anything else. It talks to a
   `StorageBackend` only. A new format means writing a new
   `StorageBackend` and (optionally) a new dequant module.
2. **The memory manager owns mapping.** Every `mmap` and every
   `munmap` goes through the `MemoryManager`, which enforces the
   cache policy and tracks peak RAM.
3. **Layer streaming is the default.** Calling `executor.step(...)`
   loads exactly one layer's tensors, runs the user callback,
   then releases them. There is no separate "load everything"
   path.

## Module layout

```text
flatrun/
├── backend/                # Format-specific storage adapters
│   ├── base.py             # StorageBackend abstract class
│   ├── safetensor.py       # Hand-written SafeTensors parser
│   ├── gguf.py             # Pure-Python GGUF v3 reader + dequant hook
│   ├── multi.py            # Composite backend (sharded checkpoints)
│   ├── registry.py         # Extension point for new formats
│   └── _populate.py        # Wires the registry on first import
├── core/                   # Type definitions shared by every layer
│   └── tensor.py           # TensorHandle / TensorView
├── runtime/                # Streaming + caching + execution
│   ├── memory.py           # MemoryManager (LRU / FIFO / byte-cap aware)
│   ├── scheduler.py        # LayerScheduler - load → compute → release
│   ├── kv_cache.py         # Per-layer KV cache
│   ├── executor.py         # ModelExecutor + StreamingExecutor
│   └── runtime.py          # InferenceRuntime - the public entry point
├── model/                  # Model-aware glue
│   ├── manifest.py         # Layer grouping
│   ├── huggingface.py      # HF checkpoint loader
│   └── qwen2.py            # Reference Qwen2 / Llama / Qwen3 forwarder
├── dequant/                # GGUF dequantisation (pure NumPy)
│   ├── gguf.py             # Q1_0, Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, Q8_0
│   ├── mlx.py              # MLX 4-bit (weight / scales / biases triples)
│   └── loader.py           # Decoder dispatch + tensor-handle hookup
├── tokenizer/              # Byte-level BPE
│   └── bpe.py              # HuggingFace tokenizer.json + GGUF metadata
├── utils/                  # Shared types, errors, memory probe
├── tests/                  # 102 unit + integration tests
└── docs/                   # This directory
```

## Request lifecycle

```text
user
  │
  ▼
InferenceRuntime.open("model.safetensors" | "model.gguf")
  │   └─► default_registry() opens the right StorageBackend
  │
  ▼
runtime.build_scheduler(layers, pre_layer, post_layer)
  │   └─► LayerScheduler keeps one layer resident at a time
  │
  ▼
executor.step(tokens)
  │
  │   for each layer in manifest:
  │       scheduler.acquire(layer) → LayerHandles (mappings live)
  │       forward_fn(layer, handles, kv_cache) → hidden
  │       scheduler.release(layer)
  │
  ▼
result.last_hidden  # (seq, vocab) logits on the final layer
```

The forwarder is the only piece that knows about the model
architecture. Everything else is format-agnostic.

## Why per-layer, not whole-model

A 70B Q4 checkpoint is ~40 GB on disk. The conventional solution
is to load the whole thing into RAM and keep it warm. FlatRun
instead treats the model file as a virtual-memory backing store:

- The scheduler mmaps one decoder block at a time and the
  memory manager evicts the previous block's mappings when
  the cache cap is exceeded.
- The KV cache lives in a separate, non-mmap region so it
  doesn't have to be paged out.
- The model file itself is opened once with `mmap`; individual
  tensors are accessed as zero-copy `np.ndarray` views into the
  file (or into a freshly `munmap`'d dequant buffer for
  quantised types).

The result is that the resident set stays bounded by the
configured cache cap (default 256 MiB) plus the KV cache, no
matter how large the model is. The cost is that the forward
pass is dominated by per-tensor `mmap` setup time on the
first access of each layer; subsequent steps are warm as long
as the cache cap is large enough to hold the working set.

## Forwarder contract

A `ForwardFn` is a plain Python callable:

```python
ForwardFn = Callable[[LayerDescriptor, LayerHandles, KVCache], np.ndarray]
```

It receives:

- `LayerDescriptor` - the manifest's view of the current layer
  (index + tensor names).
- `LayerHandles` - a dict-like view over the mmap'd handles
  for this layer only, plus a `tokens` slot the scheduler binds
  on layer 0.
- `KVCache` - the same cache the executor owns, so the
  forwarder can `kv.append(layer, k, v)` after computing them.

It returns a `np.ndarray` of shape `(seq, hidden)`. On the
final layer, the forwarder is expected to apply the final
RMSNorm and LM head and return `(seq, vocab)` logits.

The reference Qwen2 / Llama / Qwen3 forwarder in
`flatrun.model.qwen2` is a pure NumPy implementation. It is
intentionally simple: per-tensor matmuls with `np.einsum` /
`@`, no BLAS tuning, no fusing. Replacing it with a
hand-written kernel (BLAS, MLX, etc.) is the recommended
optimisation path.
