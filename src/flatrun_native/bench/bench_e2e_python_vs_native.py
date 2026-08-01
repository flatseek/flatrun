"""End-to-end benchmark: Python backend vs native backend through the forwarder.

Builds a synthetic Q4_K model and runs the forwarder with both
backends. Reports the per-decode-step wall time so we can see the
real speedup the native backend gives in the actual forwarder path
(not just the kernel in isolation).
"""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from flatrun.core.tensor import BufferTensorHandle  # noqa: E402
from flatrun.model.qwen2 import (  # noqa: E402
    Qwen2Config,
    make_qwen2_forwarder,
)
from flatrun.runtime.kv_cache import KVCache  # noqa: E402
from flatrun.runtime.scheduler import LayerHandles  # noqa: E402
from flatrun.utils.types import LayerDescriptor, TensorKey, TensorMetadata  # noqa: E402


# Qwen2-0.6B-shaped (rounded to Q4_K block boundaries).
# H=896 and INTER=4864 are not multiples of 256, so the on-disk
# Q4_K tensor is padded to (H, 1024) and (INTER, 5120). The kernel
# requires padded shapes; we use the padded dimensions directly.
H = 1024
INTER = 5120
N_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = H // N_HEADS  # 64
VOCAB = 8192
N_LAYERS = 6
PREFILL = 32
DECODE_STEPS = 4


def _q4k_block(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    out = bytearray(struct.pack("<ee", float(rng.uniform(0.01, 1.0)), float(rng.uniform(0.0, 0.5))))
    out += bytes(rng.integers(0, 64, size=12, dtype=np.uint8))
    out += bytes(rng.integers(0, 16, size=128, dtype=np.uint8))
    return bytes(out)


def make_q4k_handle(name: str, n: int, k: int, seed: int):
    # Q4_K packs blocks of 256 elements. The actual on-disk row
    # length is ``ceil(k/256) * 144`` bytes. The kernel's signature
    # accepts ``k`` as the input feature count and computes
    # ``n_blocks = k / 256`` — so the handle must report the
    # *padded* shape (k rounded up to a multiple of 256) for the
    # kernel's byte-size check to match.
    n_blocks_per_row = (k + 255) // 256
    k_padded = n_blocks_per_row * 256
    raw = bytearray()
    rng = np.random.default_rng(seed)
    for _ in range(n * n_blocks_per_row):
        raw += _q4k_block(hash((name, seed)) & 0xFFFF)
    raw = bytes(raw)
    meta = TensorMetadata(
        key=TensorKey(file="synthetic", name=name, backend="q4k"),
        shape=(n, k_padded),
        dtype="uint8",
        byte_size=len(raw),
        offset=0,
        quantization="Q4_K",
    )
    return BufferTensorHandle(meta, raw)


def make_f32_handle(name: str, arr: np.ndarray):
    meta = TensorMetadata(
        key=TensorKey(file="synthetic", name=name, backend=""),
        shape=arr.shape,
        dtype="float32",
        byte_size=arr.nbytes,
        offset=0,
    )
    return BufferTensorHandle(meta, arr.tobytes())


def build_synthetic_model(cfg: Qwen2Config) -> dict:
    rng = np.random.default_rng(0)
    h = {}
    h["model.embed_tokens.weight"] = make_f32_handle(
        "model.embed_tokens.weight",
        rng.standard_normal((VOCAB, H)).astype(np.float32) * 0.02,
    )
    h["model.norm.weight"] = make_f32_handle("model.norm.weight", np.ones(H, dtype=np.float32))
    h["lm_head.weight"] = make_q4k_handle("lm_head.weight", VOCAB, H, seed=2)
    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}"
        h[f"{prefix}.input_layernorm.weight"] = make_f32_handle(
            f"{prefix}.input_layernorm.weight", np.ones(H, dtype=np.float32)
        )
        h[f"{prefix}.post_attention_layernorm.weight"] = make_f32_handle(
            f"{prefix}.post_attention_layernorm.weight", np.ones(H, dtype=np.float32)
        )
        h[f"{prefix}.self_attn.q_proj.weight"] = make_q4k_handle(
            f"{prefix}.self_attn.q_proj.weight", H, H, seed=10 + i
        )
        h[f"{prefix}.self_attn.k_proj.weight"] = make_q4k_handle(
            f"{prefix}.self_attn.k_proj.weight", N_KV_HEADS * HEAD_DIM, H, seed=20 + i
        )
        h[f"{prefix}.self_attn.v_proj.weight"] = make_q4k_handle(
            f"{prefix}.self_attn.v_proj.weight", N_KV_HEADS * HEAD_DIM, H, seed=30 + i
        )
        h[f"{prefix}.self_attn.o_proj.weight"] = make_q4k_handle(
            f"{prefix}.self_attn.o_proj.weight", H, H, seed=40 + i
        )
        h[f"{prefix}.mlp.gate_proj.weight"] = make_q4k_handle(
            f"{prefix}.mlp.gate_proj.weight", INTER, H, seed=50 + i
        )
        h[f"{prefix}.mlp.up_proj.weight"] = make_q4k_handle(
            f"{prefix}.mlp.up_proj.weight", INTER, H, seed=60 + i
        )
        h[f"{prefix}.mlp.down_proj.weight"] = make_q4k_handle(
            f"{prefix}.mlp.down_proj.weight", H, INTER, seed=70 + i
        )
    return h


LAYER_NAMES = (
    "model.layers.{i}.input_layernorm.weight",
    "model.layers.{i}.self_attn.q_proj.weight",
    "model.layers.{i}.self_attn.k_proj.weight",
    "model.layers.{i}.self_attn.v_proj.weight",
    "model.layers.{i}.self_attn.o_proj.weight",
    "model.layers.{i}.post_attention_layernorm.weight",
    "model.layers.{i}.mlp.gate_proj.weight",
    "model.layers.{i}.mlp.up_proj.weight",
    "model.layers.{i}.mlp.down_proj.weight",
)


def drive(forwarder, handles, kv, seq_len):
    tokens = np.random.randint(0, VOCAB, size=seq_len).tolist()
    last = N_LAYERS - 1
    for i in range(N_LAYERS):
        names = tuple(n.format(i=i) for n in LAYER_NAMES)
        if i == 0:
            names = ("model.embed_tokens.weight",) + names
        if i == last:
            names = names + ("model.norm.weight", "lm_head.weight")
        lh = LayerHandles({n: handles[n] for n in names}, names)
        lh.is_first = (i == 0)
        lh.is_last = (i == last)
        lh.tokens = tokens
        lh.layer_index = i
        ld = LayerDescriptor(index=i, tensor_names=names)
        forwarder(ld, lh, kv)


def bench(name: str, fn, iters: int = 30, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    print(f"  {name:<32} median={samples[len(samples)//2]:.4f} ms")
    return samples[len(samples) // 2]


def main():
    cfg = Qwen2Config(
        hidden_size=H, intermediate_size=INTER,
        num_hidden_layers=N_LAYERS, num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS, vocab_size=VOCAB,
        rms_norm_eps=1e-6, rope_theta=1000000.0,
        max_position_embeddings=8192, tie_word_embeddings=False,
        rope_interleaved=False, sliding_window=None,
        attn_logit_softcap=None, query_pre_attn_scalar=None,
        model_arch="qwen2",
    )

    from flatrun.runtime.backend import get_backend

    # ---- Python backend ----
    handles_py = build_synthetic_model(cfg)
    fwd_py = make_qwen2_forwarder(
        cfg, enable_dequant_cache=True, backend=get_backend("python"),
    )
    kv_py = KVCache(capacity=4096)
    drive(fwd_py, handles_py, kv_py, PREFILL)  # warmup
    handles_py = build_synthetic_model(cfg)
    fwd_py = make_qwen2_forwarder(
        cfg, enable_dequant_cache=True, backend=get_backend("python"),
    )
    kv_py = KVCache(capacity=4096)
    drive(fwd_py, handles_py, kv_py, PREFILL)
    print("=== Python backend ===")
    t_py = bench("python (decode-1)", lambda: drive(fwd_py, handles_py, kv_py, 1))

    # ---- Native backend ----
    handles_native = build_synthetic_model(cfg)
    fwd_native = make_qwen2_forwarder(
        cfg, enable_dequant_cache=True, backend=get_backend("native"),
    )
    kv_native = KVCache(capacity=4096)
    drive(fwd_native, handles_native, kv_native, PREFILL)
    handles_native = build_synthetic_model(cfg)
    fwd_native = make_qwen2_forwarder(
        cfg, enable_dequant_cache=True, backend=get_backend("native"),
    )
    kv_native = KVCache(capacity=4096)
    drive(fwd_native, handles_native, kv_native, PREFILL)
    print("=== Native backend ===")
    t_native = bench("native (decode-1)", lambda: drive(fwd_native, handles_native, kv_native, 1))

    print()
    print("=" * 60)
    print(f"Python backend (decode-1): {t_py:.4f} ms")
    print(f"Native backend (decode-1): {t_native:.4f} ms")
    print(f"Speedup: {t_py / t_native:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
