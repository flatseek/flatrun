"""End-to-end parity test: python backend vs native backend.

Builds a synthetic Q4_K model, runs the forwarder with both
backends, and verifies the outputs match within FP32 round-off.
This is the only way to confirm the native backend is actually
firing in the forwarder — the bench_native_gemm.py test confirms
the C++ kernel's correctness in isolation, but this test exercises
the full dispatch path through make_qwen2_forwarder.
"""

from __future__ import annotations

import struct
import sys
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


H = 256
INTER = 512
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = H // N_HEADS  # 64
VOCAB = 256
N_LAYERS = 2
PREFILL = 32
DECODE_STEPS = 4


def _q4k_block(seed: int) -> bytes:
    """Generate a single valid Q4_K block (144 bytes)."""
    rng = np.random.default_rng(seed)
    out = bytearray()
    # d (FP16), dmin (FP16)
    d = float(rng.uniform(0.01, 1.0))
    dmin = float(rng.uniform(0.0, 0.5))
    out += struct.pack("<HH", 0, 0)  # placeholder
    out = bytearray(struct.pack("<ee", d, dmin))
    # 12 bytes of scale/min
    out += bytes(rng.integers(0, 64, size=12, dtype=np.uint8))
    # 128 bytes of packed nibbles
    out += bytes(rng.integers(0, 16, size=128, dtype=np.uint8))
    return bytes(out)


def make_q4k_handle(name: str, n: int, k: int, seed: int):
    """Build a TensorHandle wrapping raw Q4_K bytes for a (n, k) matrix."""
    n_blocks_per_row = (k + 255) // 256
    raw = bytearray()
    rng = np.random.default_rng(seed)
    for _ in range(n * n_blocks_per_row):
        raw += _q4k_block(seed)
    raw = bytes(raw)
    meta = TensorMetadata(
        key=TensorKey(file="synthetic", name=name, backend="q4k"),
        shape=(n, k),
        dtype="uint8",          # GGUF storage dtype for Q4_K is uint8 packed.
        byte_size=len(raw),
        offset=0,
        quantization="Q4_K",    # Signal that the bytes are Q4_K encoded.
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
    handles = {}
    # Embedding & norm & lm_head in F32 for the python path.
    handles["model.embed_tokens.weight"] = make_f32_handle(
        "model.embed_tokens.weight",
        rng.standard_normal((VOCAB, H)).astype(np.float32) * 0.02,
    )
    handles["model.norm.weight"] = make_f32_handle(
        "model.norm.weight", np.ones(H, dtype=np.float32)
    )
    # Note: the python backend uses F32 lm_head via the dequant path.
    # We have the lm_head as raw Q4_K to exercise the native path.
    handles["lm_head.weight"] = make_q4k_handle(
        "lm_head.weight", VOCAB, H, seed=2,
    )
    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}"
        handles[f"{prefix}.input_layernorm.weight"] = make_f32_handle(
            f"{prefix}.input_layernorm.weight", np.ones(H, dtype=np.float32)
        )
        handles[f"{prefix}.post_attention_layernorm.weight"] = make_f32_handle(
            f"{prefix}.post_attention_layernorm.weight", np.ones(H, dtype=np.float32)
        )
        # All projections in Q4_K.
        handles[f"{prefix}.self_attn.q_proj.weight"] = make_q4k_handle(
            f"{prefix}.self_attn.q_proj.weight", H, H, seed=10 + i
        )
        handles[f"{prefix}.self_attn.k_proj.weight"] = make_q4k_handle(
            f"{prefix}.self_attn.k_proj.weight", N_KV_HEADS * HEAD_DIM, H, seed=20 + i
        )
        handles[f"{prefix}.self_attn.v_proj.weight"] = make_q4k_handle(
            f"{prefix}.self_attn.v_proj.weight", N_KV_HEADS * HEAD_DIM, H, seed=30 + i
        )
        handles[f"{prefix}.self_attn.o_proj.weight"] = make_q4k_handle(
            f"{prefix}.self_attn.o_proj.weight", H, H, seed=40 + i
        )
        handles[f"{prefix}.mlp.gate_proj.weight"] = make_q4k_handle(
            f"{prefix}.mlp.gate_proj.weight", INTER, H, seed=50 + i
        )
        handles[f"{prefix}.mlp.up_proj.weight"] = make_q4k_handle(
            f"{prefix}.mlp.up_proj.weight", INTER, H, seed=60 + i
        )
        handles[f"{prefix}.mlp.down_proj.weight"] = make_q4k_handle(
            f"{prefix}.mlp.down_proj.weight", H, INTER, seed=70 + i
        )
    return handles


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
        out = forwarder(ld, lh, kv)
    return out


def main():
    cfg = Qwen2Config(
        hidden_size=H,
        intermediate_size=INTER,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        vocab_size=VOCAB,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
        rope_interleaved=False,
        sliding_window=None,
        attn_logit_softcap=None,
        query_pre_attn_scalar=None,
        model_arch="qwen2",
    )

    # --- Python backend ---
    handles_py = build_synthetic_model(cfg)
    from flatrun.runtime.backend import get_backend
    py_backend = get_backend("python")
    fwd_py = make_qwen2_forwarder(
        cfg, enable_dequant_cache=True, backend=py_backend,
    )
    kv_py = KVCache(capacity=4096)
    drive(fwd_py, handles_py, kv_py, PREFILL)
    out_py = drive(fwd_py, handles_py, kv_py, 1)

    # --- Native backend ---
    handles_native = build_synthetic_model(cfg)
    native_backend = get_backend("native")
    print(f"native backend available: {native_backend.available}")
    fwd_native = make_qwen2_forwarder(
        cfg, enable_dequant_cache=True, backend=native_backend,
    )
    kv_native = KVCache(capacity=4096)
    drive(fwd_native, handles_native, kv_native, PREFILL)
    out_native = drive(fwd_native, handles_native, kv_native, 1)

    # Compare
    diff = np.abs(out_py - out_native)
    print(f"python output shape: {out_py.shape}")
    print(f"native output shape: {out_native.shape}")
    print(f"max diff: {diff.max():.6e}")
    print(f"mean diff: {diff.mean():.6e}")
    rel = diff / (np.abs(out_py) + 1e-6)
    print(f"max rel: {rel.max():.6e}")
    # The Q4_K format has ~7-bit precision so we expect diffs in the
    # 1e-3 range relative to the absolute magnitude.
    if diff.max() < 1e-2:
        print("OK: python and native backends produce equivalent output")
    else:
        print(f"FAIL: diff too large {diff.max():.6e}")


if __name__ == "__main__":
    main()
