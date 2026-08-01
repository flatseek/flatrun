# Performance audit chain

Flatrun went through several focused performance audits during
pre-release. Each one instrumented the production path,
isolated *one* bottleneck, applied the smallest possible
fix, and re-measured. This document is the consolidated
narrative of those audits — what was found, what changed,
and what is at the pure-NumPy ceiling.

The audit scripts themselves are reproducible against
any GGUF / SafeTensors model the runtime supports. They
live outside the source tree in the developer's `/tmp/`
because each one is hundreds of lines of harness code and
isn't a long-lived diagnostic tool:

- `/tmp/projection_audit.py` — per-projection sub-stage
  breakdown (qkv/o/gateup/down: prepare, gemm, bias,
  post).
- `/tmp/attention_audit.py` — per-stage breakdown of the
  QK / AV / RoPE / KV stack / GQA repeat / softmax /
  mask chain.
- `/tmp/dequant_audit.py` — per-dequant breakdown
  (load, unpack, scales, mins, bit unpack, arithmetic,
  finish) for Q4_K / Q5_K / Q6_K / Q8_0 / F16 / F32.

Run any of them against a real model:

```bash
PYTHONPATH=src python3 /tmp/projection_audit.py
```

The headline numbers from the audit chain are:

| audit | main finding | fix | measured impact |
|-------|--------------|-----|------------------|
| Projection | `.astype(F32)` redundantly copied F32 → F32 | `.astype(np_dtype, copy=False)` (14 sites) | 4.16× per-layer projection work on 14B; -30 % top-4 projection time on 0.6B |
| Projection | `_finish` cast redundant F32 → F32 in dequant output | `.astype(F32, copy=False)` once | ~30 ms saved per 14B Q4_K_M dequant |
| MLP | `silu = gate / (1 + exp(-clip(gate))) * up` issued 5 NumPy expressions and 5 buffers | fused into one `np.empty_like` + 5 in-place ufuncs | 1.40-1.56× speedup on silu_mul stage |
| Attention | `scale = 1.0 / np.sqrt(head_dim)` (Python float) promoted F32 attention → F64 throughout the entire path including downstream MLPs | `np.float32(1.0 / np.sqrt(head_dim))` at the top of each decoder block | attention intermediates halved in bytes; downstream MLP hidden state stays F32 |
| Attention + KV | `kv.stack` called `np.stack` O(N) per decoder step, allocating fresh buffers | per-layer preallocated growing F32 buffer; `stack` is a view | 60-12,700× speedup on `stack()`; attention share 36 % → 27 % |
| Dequant cache | unbounded cache OOMs 14B hosts at ~7 GB F32 heap | `_SlidingDequantCache` with `--dequant-cache-stride N` | memory bounded; **0.9-1.2 GB/s** on Q4_K (Python-dispatch bound) — needs a compiled kernel for further speedup |

The rest of this document records *what each audit found*
in narrative form so future readers can understand the
shape of the runtime.

---

## 1. The dequant-cost era (the early state)

The starting profile of a Qwen3-0.6B Q4_K_M forward pass was:

```
Dequantization          74 %
Attention               14 %
MLP                     10 %
```

The dequant kernel itself sat at the top of every hotspot
list. Per-call Q4_K dequant of a 1.7 MB `gate_w` ran in
~19 ms in pure NumPy. Multiplying by 24 layers × 11 tokens
(prefill + decode) → ~5 sec of pure dequant. The
dequantization was dispatch-bound (Python + NumPy ufunc
per-call cost dominated over the actual arithmetic),
sitting at ~0.9-1.2 GB/s effective bandwidth — an order
of magnitude below the M-series memory ceiling.

**Fixes that landed for that era**:

- `madvise(MADV_DONTNEED)` for tensors ≥ 1 MiB on handle
  close. Without this hint the page cache held the
  touched mmap pages indefinitely and RSS grew past
  30 GB even with the cache cap honoured.
- `--dequant-cache on` defaulted to on. The unbounded
  cache keeps every decoded F32 weight alive for the
  process lifetime; for 14 B Q4_K_M this adds ~7 GB of
  F32 heap which OOMs memory-constrained hosts.
- `--dequant-cache-stride N` lets callers bound the
  cache to the last N layers (sliding window). Pre/post-
  layer tensors (embedding, norm, lm_head) carry no
  `model.layers.N.` token and are auto-immune.
- `_finish(copy=False)` in `dequant/gguf.py`. The shared
  tail of every GGUF dequant function was paying a
  redundant output-buffer memcpy on the production F32
  path. Single one-line change lifts every dequant
  (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q1_0).

**What did NOT help**:

- Fully vectorising the Q4_K dequant loop in pure
  NumPy (single broadcast across 4 groups) measured
  **0.7-0.8× the loop's throughput** because the
  transient F32 buffers (n_blocks × 256 each) cost
  more memory bandwidth than they save in dispatch. Real
  K-quant acceleration requires a compiled kernel
  (Numba / Cython / C / vDSP).
- Hoisting the type-conversion out of the loop and
  casting to F32 in advance measured **the same or
  slightly slower** — the intermediates cost more
  memory bandwidth than the dispatch overhead they
  save.

The dequant pipeline is now sitting at the ceiling for
what pure NumPy can do. Compiled kernels are the only
next step.

---

## 2. The projection-matmul era

After enabling the dequant cache the profile flipped to:

```
MLP              35-40 %
Dequantization   10-15 %
Attention        25-35 %
```

The projection matmuls were now the dominant cost. A
per-projection sub-stage audit (`/tmp/projection_audit.py`)
instruments every projection into four stages (prepare,
gemm, bias, post) and reports each stage's time,
allocation, and the layout of every intermediate.

Findings:

- The `.astype(np.float32)` before each matmul was
  defaulting to `copy=True`, so even when the weight
  was already F32 it allocated a fresh buffer. On
  Qwen3-14B Q4_K_M, `gate_w.astype(F32)` alone measured
  ~107 ms per call. Across all 4 projections × 24
  layers × 11 tokens, that's 51 sec of pure memcpy
  waste per inference.

  **Fix**: replace each `.astype(np.float32)` on the
  weight (and on the matmul output, in `_compute_layer_logits`)
  with `.astype(np_dtype, copy=False)`. The 14 hot-path
  sites (qkv/o/gateup/down across all three decoder
  blocks + LM head + analyzer) all share the same
  pattern. Per-layer projection work dropped **270 ms
  → 65 ms** in the 14 B synthetic benchmark (4.16×);
  end-to-end top-4 projection time on Qwen3-0.6B
  dropped ~30 %.
- All weight transposes were already zero-copy views
  (`weight.T`, `OWNDATA=False`); the F-strided view
  reads naturally via Accelerate's `cblas_sgemm`
  `transB=T` flag.
- All output splits (q/k/v from qkv, gate/up from
  gateup) were already zero-copy views.
- The `np.concatenate` for `qkv_w` and `gateup_w`
  was — and still is — being re-allocated every
  decoder step (cache-on or not). Per call:
  ~16 MB on 0.6B, ~370 MB on 14B. **Not addressed in
  this release** — caching the concat is the next pure-
  NumPy item on the optimization table (estimated
  ~8-12 % total runtime).
- The `silu = gate / (1 + exp(-clip(gate))) * up`
  chain issued 5 NumPy expressions and allocated 5
  intermediate F32 buffers. **Fix**: fused into a
  single `np.empty_like(gate)` + 5 in-place ufuncs
  (`np.exp` → `+ 1.0` → `np.reciprocal` → `np.multiply`
  → `np.multiply`). Measured **1.40-1.56× speedup** on
  the post-gateup stage. The clip was dropped because
  IEEE-754 gives the correct asymptotics at ±88.

After this audit cycle, projections are 85-97 % BLAS
time per the sub-stage audit. The only remaining pure-
NumPy work in the projection path is the `np.concatenate`
for qkv_w / gateup_w — caching those is the easy next
win.

---

## 3. The attention-matmul era

After the projection fix:

```
MLP                 36-44 %
Dequantization      17-27 %
Attention           19-30 %
```

`/tmp/attention_audit.py` broke attention into eleven
sub-stages (q/k norm, RoPE prepare/compute/post, KV
append, KV stack, GQA repeat, QK gemm, QK mask, softmax,
AV gemm) and reported each stage's time, allocation,
and the layout of every intermediate.

Two concrete bugs were found and fixed:

1. **F64 leak via Python-float scale** at the top of
   every decoder block. `1.0 / np.sqrt(head_dim)` is a
   Python `float` (float64); when multiplied by an F32
   einsum result, NumPy's NEP-50 promoted the entire
   result to float64. The F64 leaked through:
   - `attn + causal_mask` → still F64
   - softmax inputs → F64
   - `einsum("htT,Thd->thd", attn, v_full)` → F64 output
   - reshape → F64 view
   - `attn_out @ o_w.T` → F64 result
   - **The residual `hidden = residual + attn_out`**
     → hidden state becomes F64
   - Every subsequent RMSNorm, MLP matmul, and softmax
     ran on F64 hidden state — effectively doubling
     every byte touched downstream of attention.

   **Fix**: `scale = np.float32(1.0 / np.sqrt(head_dim))`
   at the top of each decoder block
   (`_decoder_block`, `_gemma3_decoder_block`,
   `_qwen35_full_attention_block`) and in the final
   norm + LM head path inside `forward()`. Verified
   end-to-end via the audit:
   `attn_pre_mask` dtype `float64` nbytes 15.1 KB →
   `float32` nbytes 7.6 KB.

2. **`kv.stack` allocates `O(T)` per decoder step.** The
   previous `KVCache` stored past K/V as a Python list
   of per-token `(h, d)` arrays and rebuilt the full
   history via `np.stack` on every `stack(layer)` call.
   At `past_len=120` the audit measured `kv_stack`
   at 645 µs — **32 % of attention share** — and the
   cost grows as `O(T)` from a single `np.stack`
   dispatch.

   **Fix**: rewrite `KVCache` around a per-layer
   preallocated growing F32 buffer (`_LayerKV`). Each
   layer owns `(cap, kv_heads, head_dim)`; `cap *= 2`
   on overflow (amortised O(1) per append); `stack` is
   a slice view. Measured isolated speedup on
   `stack()` alone:

   | past | old (`np.stack`) | new (view) | speedup |
   |-----:|------------------:|-----------:|--------:|
   | 16 | 26.1 µs | 0.42 µs | 61× |
   | 64 | 91.7 µs | 0.42 µs | 217× |
   | 256 | 393.2 µs | 0.49 µs | 803× |
   | 1024 | 2268 µs | 0.72 µs | 3165× |
   | 4096 | 8365 µs | 0.66 µs | **12,704×** |

   End-to-end attention share fell from ~36 % to ~27 %
   on Qwen3-0.6B Q4_K_M with `--max-new 10`.

After this audit cycle, the attention profile looks like:

| sub-stage | share | notes |
|-----------|------:|-------|
| `qk_matmul` | 9-15 % | Accelerate sgemm dispatch via einsum |
| `av_matmul` | 7-17 % | same |
| `rope` | 9 % | cos/sin lookup + 2 broadcast mults |
| `softmax` | 5-7 % | 4 intermediate F32 buffers |
| `qk_mask` | 7-29 % | F32 mask add on every step |
| `kv_stack` | < 1 % | zero-copy view (was 32 % at past=120) |
| `gqa_repeat` | 13 % | `np.repeat(k_hist, head_group, axis=1)` materialises a head_group× replicated buffer — not yet optimised |
| `q/k rms_norm` | 6-8 % | per-head RMSNorm before RoPE |

Remaining pure-NumPy attention items:

- `gqa_repeat` `np.repeat` → strided broadcast view
  (validates against Accelerate sgemm dispatch);
  expected ~5-10 % attention share.
- `qk_mask` → fuse into softmax with `np.where`; ~5 %.
- RoPE → in-place chain (similar to the SiLU fix
  applied in the MLP era); ~30-50 % of RoPE itself,
  ~3-4 % of attention share.

---

## 4. The dequant-residual era

After the attention fixes, the profile flipped again:

```
MLP              41 %
Dequantization   35 %
Attention        19 %
```

The dequantization bucket is now the single largest
share again. `/tmp/dequant_audit.py` broke each dequant
function into per-stage sub-timings (load, unpack, scales,
mins, bit unpack, arithmetic, finish, reshape, cache
lookup, cache insert) and reported each stage's
allocation, layout, and contiguity.

Findings:

- Q4_K is **Python-dispatch bound**. The 4-iteration
  Python loop and the per-iteration `(chunk & 0x0F)
  .astype(F32)`, `(chunk >> 4).astype(F32)` allocate
  ~9.4 MB of intermediates per call (8 buffers × 50 KB
  on 0.6B; 8 × 75 KB on 14B).
- Effective bandwidth: **0.9-1.2 GB/s** for Q4_K on the
  M-class, vs the ~30-40 GB/s ceiling of unified
  memory — an order of magnitude under.
- The cache efficiency is essentially perfect: with
  `--dequant-cache on` (default), every decoded
  weight hits cache on the second-and-later decode
  steps; per-call dict lookup overhead is ~1 µs and
  doesn't move the needle.

What pure NumPy can still do:

- Cache the `np.concatenate` for `qkv_w` and
  `gateup_w` (extension of the sliding cache in
  `--dequant-cache-stride`); ~8-12 % total runtime.
- Replace `gqa_repeat` `np.repeat` with a strided
  broadcast view; ~5-10 % attention.

What pure NumPy **cannot** fix:

- The Q4_K 4-iteration loop. Fully-vectorising it in
  pure NumPy measures **0.7-0.8× the loop's
  throughput** because the intermediates cost more
  memory bandwidth than they save in dispatch.

What needs a compiled kernel:

- Numba JIT for the Q4_K inner loop; typical reported
  gains: 3-10× over pure NumPy.
- Quantized matmul (skip dequant entirely; do `int4 @
  fp16` directly). Requires a custom kernel operator.
- FP16 weight storage (halve the dequant output bytes
  and the matmul reads). Pure NumPy feasible on the
  dtype plumbing side; expected 1.5-2× on memory-
  bound decoder paths.

---

## Per-stage audit summary (one-shot table)

Single-call measurements on Qwen3-0.6B Q4_K_M, recorded
once per audit cycle. Numbers are absolute `ms` per
projection per decode step.

| bucket | sub-stage | before | after first fix | after second fix |
|--------|-----------|------:|----------------:|-----------------:|
| qkv | prepare | ~5 ms | ~0.16 ms | ~0.05 ms |
| qkv | gemm (BLAS) | ~3 ms | ~3 ms | ~0.9 ms |
| qkv | post | ~5 ms | ~0.18 ms | ~0.18 ms |
| gateup | prepare | ~5 ms | ~0.07 ms | ~0.07 ms |
| gateup | gemm (BLAS) | ~5 ms | ~5 ms | ~1.4 ms |
| gateup | post | ~5 ms | ~0.36 ms | ~0.36 ms (then 0.23 via SiLU fuse) |
| down | gemm (BLAS) | ~3 ms | ~0.5 ms | ~0.5 ms |
| o | gemm (BLAS) | ~1 ms | ~0.5 ms | ~0.5 ms |
| attention | qk_matmul | ~1 ms | ~0.5 ms | ~0.5 ms |
| attention | av_matmul | ~3 ms | ~0.3 ms | ~0.3 ms |
| attention | kv_stack | ~0.6 ms (past=120) | ~0.6 ms | < 0.01 ms (view) |
| dequantization | per-weight avg | ~20 ms | ~3 ms (cache warm) | ~3 ms |

(Bench conditions vary between audit cycles; the
absolute numbers are illustrative. The relative win
ratios are stable.)

---

## Realistic remaining pure-NumPy ceiling

If every remaining item ships:

| # | item | est. additional total-runtime speedup | complexity |
|--:|------|---------------------------------------:|------------|
| 1 | cache `qkv_w` and `gateup_w` `np.concatenate` | 8-12 % | Low |
| 2 | replace `gqa_repeat` `np.repeat` with strided broadcast view | 5-10 % attention share | Medium |
| 3 | fuse RoPE into in-place chain | 3-4 % attention share | Low |
| 4 | fuse mask add into softmax with `np.where` | ~5 % attention share | Low |
| 5 | FP16 weight storage (dtype plumbing + memory cap) | 1.5-2× on memory-bound decoder paths | Medium |
| **Total achievable without compiled kernels** | | **~15-20 %** | |

Beyond that, the only path to a structural speedup is a
compiled kernel rewrite (Numba / Cython / MPS) of the
dequant kernel + the matmul + the fused operations.

---

## Where the bottleneck *won't* move to

After all of the above lands, the remaining share will be:

- **Accelerate GEMM dispatch cost** itself — irreducible
  in pure NumPy.
- **NumPy ufunc dispatch cost** — the cost of every
  `np.exp`, `np.multiply`, `np.add` call into the C
  dispatcher.
- **Python interpreter overhead** — the per-call cost of
  every Python function frame.

These three together are what remain once the algorithmic
and allocation issues are fixed. They are the fundamental
ceiling of "Python + NumPy + Apple Accelerate" for this
workload.

To break that ceiling, the work moves out of Python
entirely: Numba / Cython / MPS, with all the maintenance
cost that entails.
