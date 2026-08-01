# Performance review of the forward pass

This is a critical, evidence-based review of the forward-pass path
in `flatrun.model.qwen2`. The focus is *latency*, not accuracy. The
review is the basis for the `--profile-detailed` instrumentation
that ships in this release.

The review reads the code top-to-bottom and identifies *real*
bottlenecks — places where the implementation is doing more work
than the math requires, with an estimated cost. Speculative
"you could use BLAS" advice is kept until the end.

When this review was written, the runtime went through the
following forward pass for each layer:

1. Load per-layer weights via the scheduler.
2. RMSNorm on the residual input.
3. `Q @ q_w.T + K @ k_w.T + V @ v_w.T` collapsed into a single
   `X @ qkv_w.T` after concatenating the three weights.
4. Reshape QKV to (seq, n_heads, head_dim) / (seq, n_kv_heads, head_dim).
5. (Qwen3) RMSNorm on Q and K per head.
6. RoPE on Q and K.
7. Python loop over `seq_len` appending K/V to the KV cache.
8. Re-stack the KV cache.
9. `np.repeat` for GQA expansion.
10. `einsum("thd,Thd->htT")` for QK matmul.
11. `_causal_mask(seq_len, past_len)` allocated fresh.
12. Softmax.
13. `einsum("htT,Thd->thd")` for AV matmul.
14. `O @ o_w.T` for the output projection.
15. Residual add.
16. MLP RMSNorm.
17. `Gate @ gate_w.T + Up @ up_w.T` collapsed into `X @ gateup_w.T`.
18. SiLU + elementwise multiply.
19. `Down @ down_w.T`.
20. Residual add.

Each step is examined below.

---

## 1. The QKV concatenation + astype (BIGGEST CHURN)

```python
q_w, q_b = _fetch_linear_with_quant(...)
k_w, k_b = _fetch_linear_with_quant(...)
v_w, v_b = _fetch_linear_with_quant(...)
q_w = _as_linear(q_w, config.hidden_size, ...)
k_w = _as_linear(k_w, config.hidden_size, ...)
v_w = _as_linear(v_w, config.hidden_size, ...)
qkv_w = np.concatenate([q_w, k_w, v_w], axis=0).astype(np.float32)
qkv = x @ qkv_w.T
```

For a 14B Qwen3 with `hidden_size=5120`, `q_dim=8192`, `kv_dim=512`:

| Buffer | Shape | Bytes (fp32) |
|---|---|---|
| `q_w` | (8192, 5120) | 160 MB |
| `k_w` | (512, 5120) | 10 MB |
| `v_w` | (512, 5120) | 10 MB |
| `qkv_w` (concat) | (8704, 5120) | 178 MB |
| `.astype(np.float32)` | (8704, 5120) | another 178 MB if dtype != fp32 |

**Allocation cost per layer**: ~360 MB written, plus the previous
layer's buffer torn down. Across 40 layers, that's ~14 GB of
allocation churn per forward pass. The memory manager has to
evict/load around this churn even though the *useful* total is
~180 MB.

The `.astype(np.float32)` is the worst part: when the dequant
output is fp16 or bf16 (MLX-4bit, fp16 SafeTensors), NumPy must
*allocate a new buffer* and copy every byte. The intermediate
fp32 buffer is then held for the duration of the matmul.

**Why the code is structured this way**: one fused matmul is
faster than three sequential matmuls because of BLAS/LAPACK
cache locality. The concat-then-matmul pays 178 MB of copy to
save a few % of matmul time.

**Estimated impact**: 12-25% of total inference time on large
models where the concat/astype dwarfs the matmul itself.
*Highest-impact optimisation.*

**Suggested fix**: pre-allocate a single fp32 buffer that fits
`max(q_dim + 2*kv_dim, hidden_size) * hidden_size` floats and
copy into it. Or — for the default case where `dequant_cache` is
*off* — keep the dequantized fp32 weights in the cache so the
concat is over already-fp32 buffers. The biggest win is to
have the cache default to **on** for Q4_K_M / Q4_0 models where
the dequant cost is comparable to the matmul.

---

## 2. The Python loop for KV cache append (`for pos in range(seq_len)`)

```python
for pos in range(seq_len):
    kv.append(idx, k[pos], v[pos])
```

Each iteration:

1. Calls `kv.append(...)` (Python call).
2. Allocates a `KVEntry` dataclass.
3. Calls `np.asarray(k, v)` (no-op when already an ndarray, but
   the call itself has overhead).

For `seq_len=512`, that's 512 × 2 = 1024 Python iterations per
layer. For 40 layers that's 40,960 iterations per forward pass.

**Estimated impact**: 1-3% on its own.
**Why it matters**: the per-position allocation of `KVEntry`
forces the cache to grow in a non-contiguous pattern; `kv.stack`
later has to `np.stack` everything into a single contiguous
buffer anyway.

**Suggested fix**: extend the `KVCache` API to accept a whole
`k, v` pair (`kv.extend(idx, k, v)`) and slice internally on
`stack`. The Python loop disappears.

---

## 3. `_causal_mask` allocated every layer

```python
attn = attn + _causal_mask(seq_len, past_len)
```

`_causal_mask` allocates a fresh `(seq_len, total)` float32 array
filled with `-inf`, then sets the upper triangle to 0. For
`seq_len=512` that's 1 MB per layer per forward pass. Across 40
layers that's 40 MB of allocations that get freed immediately.

**Estimated impact**: <1% on its own.
**Why it matters**: allocation pressure and page-cache churn.

**Suggested fix**: cache the masks per `(seq_len, past_len)` pair;
or, since the mask only depends on the position-difference, use a
broadcast trick (`attn + (positions[:, None] < positions[None, :]) *
-inf`) that avoids the temp allocation.

---

## 4. The MLP `gateup` concatenation + astype (same as QKV)

```python
gateup = x @ np.concatenate([gate_w, up_w], axis=0).astype(np.float32).T
```

Identical pattern to the QKV case. For a 14B model with
`intermediate=18944`:

| Buffer | Shape | Bytes (fp32) |
|---|---|---|
| `gate_w` | (18944, 5120) | 370 MB |
| `up_w` | (18944, 5120) | 370 MB |
| `gateup_w` (concat) | (37888, 5120) | 740 MB |
| `.astype(np.float32)` | (37888, 5120) | another 740 MB if dtype != fp32 |

**Estimated impact**: 5-10% of total inference time. The MLP
matmul is ~2x the size of the QKV matmul, and the concat/astype
scales linearly.

**Suggested fix**: same as QKV — pre-allocate a buffer, or use
two separate matmuls and concatenate the *output* on the
seq_len axis.

---

## 5. The `np.einsum` overhead

`np.einsum("thd,Thd->htT", q, k_full) * scale` has a small
overhead relative to `np.matmul` because einsum evaluates the
contraction abstractly and dispatches to BLAS. For tensors of
this size the difference is negligible (<1%) but the pattern
shows up three times per layer (QK, AV, and the contraction for
the einsum parsing).

**Estimated impact**: <1%.
**Suggested fix**: replace with `q @ k_full.swapaxes(1, 2)` —
the BLAS dispatch is identical but the call overhead is lower.

---

## 6. The `.astype(np.float32)` on every load

```python
attn_norm_w = handles[name].as_numpy().astype(np.float32)
```

For bfloat16 weights (the default for HuggingFace SafeTensors),
`.astype(np.float32)` allocates a new buffer and widens the
elements. For float32 weights (GGUF Q8_0), the `.astype` is a
no-op (returns self) but the call still has Python overhead.

**Estimated impact**: <1% per call site. Many call sites mean
the cumulative cost is meaningful.

**Suggested fix**: check `metadata.dtype` first and skip the
astype when already float32. Add a `_should_widen()` helper.

---

## 7. The GQA replication `np.repeat`

```python
k_full = np.repeat(k_hist, head_group, axis=1)
v_full = np.repeat(v_hist, head_group, axis=1)
```

`head_group` is typically 8 (Qwen3 14B has 28 query heads and 4
KV heads). The repeat allocates an 8× larger buffer but the
matmul cost itself is unchanged.

**Estimated impact**: <0.5% (allocation only).
**Why it matters**: the expanded KV buffer is held for the
duration of the attention matmul.

**Suggested fix**: skip the repeat and use the GQA-aware matmul
directly. This is a kernel-level change and not worth the
complexity for the current pure-NumPy setup.

---

## 8. The autoregressive decode loop re-encodes the prompt

```python
for nxt in range(max_new):
    ids = prompt_ids + generated
    result = executor.step(tokens=ids)
```

The executor re-runs the entire prompt + generated sequence each
step, then `kv.reset()` clears the cache for the next step. Total
work is `O(max_new² / 2 + max_new * prompt_len)` instead of
`O(max_new)`.

For `max_new=128`, `prompt_len=500`: 128·500 + 128²/2 ≈ 72k
tokens. Versus incremental decode: 128 tokens.

**Estimated impact**: this is the largest single inefficiency
in the codebase. For typical interactive usage (max_new=128)
the autoregressive loop dominates end-to-end latency by an
order of magnitude.

**Suggested fix**: implement incremental decode in the
executor. The forwarder emits `(seq_len, hidden)` outputs that
the executor can cache; subsequent steps only process the new
token. The KV cache already supports this; the issue is that
`step()` resets the cache before each call.

**Why this is hard**: the current forwarder applies the final
norm + LM head at the *last* layer and returns logits for the
whole sequence. For incremental decode we need logits for the
*last* position only. The forwarder already returns the right
shape; the executor just needs to slice.

---

## 9. Memory accounting summary

Per forward pass through a 14B model with `seq_len=512`:

| Source | Bytes allocated | Per layer | 40 layers |
|---|---|---|---|
| QKV concat + astype | ~360 MB | ~360 MB | ~14 GB churn |
| MLP gateup concat + astype | ~740 MB | ~740 MB | ~29 GB churn |
| q/k/v/o/gate/up/down dequant | ~600 MB | ~600 MB | ~24 GB churn |
| `np.repeat` for GQA | ~256 KB | ~256 KB | ~10 MB |
| `_causal_mask` | ~1 MB | ~1 MB | ~40 MB |
| KV cache stack | ~210 MB | ~210 MB | ~8 GB churn |
| **Total non-matmul allocation** | | | **~75 GB churn** |

The matmuls themselves do ~`700 GFLOPs` total per forward pass
(roughly 2 × N_params × seq_len, scaled by the matmul
constant). On a typical laptop CPU at 30 GFLOPs/s for fp32, the
matmul alone takes ~25 seconds. The allocation churn is *not*
on the critical path for throughput once the memory manager
settles, but **it is on the critical path for memory pressure**
on memory-constrained hosts — the cache cap has to evict
buffers we just wrote.

---

## 10. The `q/k/v` reshape after the matmul

```python
q = qkv[:, :q_dim].reshape(seq_len, n_heads, head_dim)
k = qkv[:, q_dim : q_dim + kv_dim].reshape(seq_len, n_kv_heads, head_dim)
v = qkv[:, q_dim + kv_dim :].reshape(seq_len, n_kv_heads, head_dim)
```

`qkv[:, :q_dim]` is a view. `reshape` is a view because the
slice is contiguous (the matmul output is row-major).

`qkv[:, q_dim : q_dim + kv_dim]` is **not** contiguous. The
`reshape` to `(seq_len, n_kv_heads, head_dim)` therefore does a
copy.

**Estimated impact**: <0.5% (small allocations).
**Suggested fix**: do the matmul with a custom output stride that
lays out the channels as `(seq, n_kv, head_dim)` directly. Or
split the QKV matmul into three so the K and V outputs are
contiguous.

---

## 11. The hidden state is never re-allocated

`state["hidden"]` is updated by replacement, not in-place. The
`del` block at the end of `_decoder_block` drops every reference
to the previous layer's intermediates so the GC can reclaim them.

**Verdict**: this is correct. The hidden state is a single
buffer reused across layers.

---

## 12. The KV cache growth is O(N²) over a generation

`kv.stack` copies all past K/V entries into a fresh array on
every call. For a 128-token generation this is `O(128²)` per
layer. The cost is bounded by the cache capacity but for a
4096-token context that's 16 MB of K+V per layer per step.

**Estimated impact**: <1% on its own, but interacts with the
loop in item 2.

**Suggested fix**: maintain the KV cache as a pre-allocated
slab that grows by doubling. The current implementation grows
the Python list by one entry per `append()` and the
`stack()` call copies everything.

---

## 13. Things that are NOT bottlenecks

| Operation | Why it's fine |
|---|---|
| `handles[name].as_numpy()` | Returns a view of the mmap; zero copy. |
| `np.concatenate` for the q/k/v *outputs* | Small (seq_len × hidden); negligible. |
| `_apply_rope` | Vectorised over `seq_len`; small fixed cost. |
| `_softmax` | Vectorised; small fixed cost. |
| The `del` block at the end | Free; runtime help. |
| `gc.collect()` at the end | One-shot; only on `dequant_cache is None`. |

---

## 14. The proposed profiler

The release ships `--profile-detailed`, which brackets every
operation in the decoder block with a microsecond-precision
context manager. The output is a per-layer breakdown plus a
summary that aggregates by category (Attention, MLP, Tensor
Loading, Dequantization, Norm, Sampling, Other).

To use:

```bash
flatrun run --profile-detailed --profile-save profile.json \
    --prompt "halo" --model /path/to/14B --max-new 1
```

The output looks like:

```
Layer 0
---------------------------------
...

=========================
PROFILE SUMMARY
=========================
  Attention             62.4 %
  MLP                  28.1 %
  Tensor Loading        4.2 %
  Dequantization        3.1 %
  Norm                  1.5 %
  Other                 0.7 %

Top 20 Slowest Operations
  1. qkv_proj: 18.2 ms total
  2. gateup_proj: 14.7 ms total
  ...
```

The numbers tell you *where* the time actually goes. The fix
priorities in section 1, 4, and 8 are the ones that move the
needle.
