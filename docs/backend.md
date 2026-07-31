# Storage backends

A `StorageBackend` is the only thing FlatRun's runtime talks to
when it needs a tensor. Backends are registered through
`flatrun.backend.registry.default_registry()`; the shipped
implementations wire themselves in on first import.

## Contract

```python
class StorageBackend(abc.ABC):
    name: str
    supports_mmap: bool

    def open(self) -> None: ...
    def close(self) -> None: ...
    def list_tensors(self) -> Iterator[TensorKey]: ...
    def open_handle(self, name: str) -> TensorHandle: ...
```

* `open` is called once before any other method. It must
  succeed for the backend to be usable.
* `list_tensors` returns every tensor the backend exposes, in
  whatever order the backend chooses.
* `open_handle` returns a `TensorHandle` for the named tensor.
  The handle must be cheap to construct; reading bytes happens
  lazily inside `handle.view()`.

A `TensorHandle` exposes:

```python
class TensorHandle(abc.ABC):
    @property
    def metadata(self) -> TensorMetadata: ...   # shape, dtype, byte_size
    def view(self) -> TensorView: ...           # zero-copy NumPy view
    def materialize(self) -> bytes: ...         # eager byte copy
    def close(self) -> None: ...
```

If the handle wraps a quantised GGUF tensor, `metadata.dtype`
is `uint8` and `metadata.quantization` names the GGML type
(`Q4_0`, `Q4_K`, `Q8_0`, ...). The forwarder resolves those
through `flatrun.dequant.dequant_handle`.

## SafeTensors

`SafeTensorBackend` parses the JSON header itself, mmap's the
file once on `open`, and hands out `MmapTensorHandle`s that
are zero-copy views into the mapping. The runtime never has
to read the full file into RAM.

Native dtypes supported: `F32`, `F16`, `BF16`. Quantised
shards (MLX 4-bit, etc.) are exposed as `uint8` bytes; the
dequant module turns them back into floats at fetch time.

## GGUF

`GGUFBackend` reads GGUF v3 metadata and the tensor info table
in pure Python. It supports:

- Tensor name translation (Qwen3, Llama, etc.) so the same
  manifest builder that works for SafeTensors also works for
  GGUF.
- `bf16`, `f16`, `f32` tensors.
- The full set of quant types in `flatrun.dequant.gguf`:
  `Q1_0`, `Q4_0`, `Q5_0`, `Q5_1`, `Q4_K`, `Q5_K`, `Q6_K`,
  `Q8_0`.

GGUF's tensor dimensions are stored *fastest-axis-first*
(`ne[0]` is the contiguous one); the backend reverses them
on read so `metadata.shape` matches the NumPy row-major
convention. Skipping that reversal reinterprets every flat
buffer under the wrong stride - the same buffer, shape-legal
all the way to the logits, just scrambled.

### Custom quants

Some forks of llama.cpp add new GGML type ids. FlatRun ships
a small type table; if you hit `Unknown(N)` while loading a
file, add the type to `_GGML_TYPES` in
`flatrun.backend.gguf` and a matching decoder to
`flatrun.dequant.gguf` (and register it in
`flatrun.dequant.loader._GGUF_DECODERS`).

## MultiBackend

`MultiBackend` composes several backends under one name, so
sharded checkpoints (multiple `model-00001-of-00030.safetensors`
files) appear as a single backend. The runtime is
shard-agnostic: it just sees one big list of tensors and asks
for handles by name.

## Adding a new format

1. Implement `StorageBackend` (typically a few hundred lines).
2. Register it in `flatrun.backend._populate.populate` with
   the suffix / filename you want it to handle.
3. If the format uses an exotic quant, add a decoder under
   `flatrun.dequant` and wire it into `_GGUF_DECODERS`.
4. Add a unit test that round-trips a synthetic file through
   the backend.

The runtime will pick up the new format automatically; no
changes to `flatrun.runtime` are needed.
