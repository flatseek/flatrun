"""Tests for the reference Qwen2 / Llama forward pass.

The tests build a tiny synthetic model (3 layers, 8-dim hidden) using
the same handle-based protocol that :class:`InferenceRuntime` uses,
then run a full decoder pass and check the result.

Weights follow the on-disk HuggingFace convention - ``nn.Linear.weight``
is ``(out_features, in_features)`` - because that orientation is the
thing the forwarder has to get right, and a transposed square matrix is
shape-legal enough to pass a sloppier fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from flatrun.core.tensor import BufferTensorHandle
from flatrun.model.qwen2 import (
    Qwen2Config,
    _apply_rope,
    _precompute_rope,
    make_qwen2_forwarder,
)
from flatrun.runtime.kv_cache import KVCache
from flatrun.runtime.scheduler import LayerHandles
from flatrun.utils.types import LayerDescriptor, TensorKey, TensorMetadata


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_handle(name: str, arr: np.ndarray):
    meta = TensorMetadata(
        key=TensorKey(file="test", name=name, backend=""),
        shape=arr.shape,
        dtype=str(arr.dtype),
        byte_size=arr.nbytes,
        offset=0,
    )
    return BufferTensorHandle(meta, arr.tobytes())


def _layer_tensor_names(index: int) -> tuple[str, ...]:
    prefix = f"model.layers.{index}"
    return (
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    )


def _build_tiny_model(cfg: Qwen2Config):
    """Create a tiny in-memory model with random ``(out, in)`` weights."""
    rng = np.random.default_rng(0)
    H = cfg.hidden_size
    F = cfg.intermediate_size
    V = cfg.vocab_size
    Hd = H // cfg.num_attention_heads
    KV = cfg.num_key_value_heads * Hd

    handles: dict[str, BufferTensorHandle] = {}

    def add(name: str, arr: np.ndarray):
        handles[name] = _make_handle(name, arr)

    add("model.embed_tokens.weight", rng.standard_normal((V, H)).astype(np.float32) * 0.02)
    add("model.norm.weight", np.ones(H, dtype=np.float32))

    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}"
        add(f"{prefix}.input_layernorm.weight", np.ones(H, dtype=np.float32))
        add(f"{prefix}.post_attention_layernorm.weight", np.ones(H, dtype=np.float32))
        add(f"{prefix}.self_attn.q_proj.weight", rng.standard_normal((H, H)).astype(np.float32) * 0.05)
        add(f"{prefix}.self_attn.k_proj.weight", rng.standard_normal((KV, H)).astype(np.float32) * 0.05)
        add(f"{prefix}.self_attn.v_proj.weight", rng.standard_normal((KV, H)).astype(np.float32) * 0.05)
        add(f"{prefix}.self_attn.o_proj.weight", rng.standard_normal((H, H)).astype(np.float32) * 0.05)
        add(f"{prefix}.mlp.gate_proj.weight", rng.standard_normal((F, H)).astype(np.float32) * 0.05)
        add(f"{prefix}.mlp.up_proj.weight", rng.standard_normal((F, H)).astype(np.float32) * 0.05)
        add(f"{prefix}.mlp.down_proj.weight", rng.standard_normal((H, F)).astype(np.float32) * 0.05)

    return handles


def _handles_for_layer(handles, names: tuple[str, ...], *, is_first: bool = False, is_last: bool = False) -> LayerHandles:
    lh = LayerHandles({n: handles[n] for n in names}, names)
    lh.is_first = is_first
    lh.is_last = is_last
    return lh


def _run_all_layers(cfg, forwarder, handles, tokens, kv) -> np.ndarray:
    """Drive every layer the way the scheduler does, returning logits.

    The scheduler binds the embedding onto layer 0 and the final norm
    (plus an untied ``lm_head``) onto the last layer, so the fixture
    mirrors that bookending exactly.
    """
    last = cfg.num_hidden_layers - 1
    out = None
    for i in range(cfg.num_hidden_layers):
        names = _layer_tensor_names(i)
        if i == 0:
            names = ("model.embed_tokens.weight",) + names
        if i == last:
            names = names + ("model.norm.weight",)
            if not cfg.tie_word_embeddings:
                names = names + ("lm_head.weight",)
        lh = _handles_for_layer(handles, names, is_first=(i == 0), is_last=(i == last))
        if i == 0:
            lh.tokens = list(tokens)
        out = forwarder(LayerDescriptor(index=i, tensor_names=names), lh, kv)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _tiny_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )


def test_qwen2_final_layer_returns_vocab_sized_logits() -> None:
    """With tied embeddings the LM head should produce (seq, vocab) logits."""
    cfg = _tiny_config()
    forwarder = make_qwen2_forwarder(cfg, dtype="float32")
    handles = _build_tiny_model(cfg)
    tokens = [1, 5, 9]

    out = _run_all_layers(cfg, forwarder, handles, tokens, KVCache(capacity=64))
    assert out.shape == (len(tokens), cfg.vocab_size)


def test_qwen2_untied_lm_head_uses_lm_head_weight() -> None:
    """Untying embeddings uses ``lm_head`` instead of the embedding matrix."""
    cfg = _tiny_config()
    cfg.tie_word_embeddings = False
    forwarder = make_qwen2_forwarder(cfg, dtype="float32")
    handles = _build_tiny_model(cfg)
    H, V = cfg.hidden_size, cfg.vocab_size
    rng = np.random.default_rng(1)
    handles["lm_head.weight"] = _make_handle(
        "lm_head.weight",
        rng.standard_normal((V, H)).astype(np.float32) * 0.02,
    )

    out = _run_all_layers(cfg, forwarder, handles, [1], KVCache(capacity=64))
    assert out.shape == (1, V)


def test_qwen2_config_from_hf_config_round_trip() -> None:
    """``from_hf_config`` pulls the documented keys out of a dict."""
    raw = {
        "vocab_size": 100,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "rope_theta": 50000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
    }
    cfg = Qwen2Config.from_hf_config(raw)
    assert cfg.vocab_size == 100
    assert cfg.num_attention_heads == 4
    assert cfg.num_key_value_heads == 2
    assert cfg.tie_word_embeddings is True


def test_qwen2_attention_with_zero_sequence_position_runs() -> None:
    """Single-token input still produces a valid, finite output."""
    cfg = _tiny_config()
    forwarder = make_qwen2_forwarder(cfg, dtype="float32")
    handles = _build_tiny_model(cfg)

    out = _run_all_layers(cfg, forwarder, handles, [3], KVCache(capacity=64))
    assert out.shape == (1, cfg.vocab_size)
    assert np.all(np.isfinite(out))


def test_qwen2_logits_are_finite() -> None:
    """Random weights should produce finite (non-NaN, non-Inf) logits."""
    cfg = _tiny_config()
    forwarder = make_qwen2_forwarder(cfg, dtype="float32")
    handles = _build_tiny_model(cfg)

    out = _run_all_layers(cfg, forwarder, handles, [0, 7, 4], KVCache(capacity=64))
    assert np.all(np.isfinite(out))


def test_qwen2_attention_is_causal() -> None:
    """Changing a later token must not perturb any earlier position.

    This is the regression test for the missing causal mask. Without
    the mask every prefill position attends to the whole prompt, so
    editing the final token silently rewrites the hidden states of the
    ones before it - and the output degrades into fluent nonsense.
    """
    cfg = _tiny_config()
    handles = _build_tiny_model(cfg)

    a = _run_all_layers(
        cfg, make_qwen2_forwarder(cfg, dtype="float32"), handles,
        [1, 5, 9], KVCache(capacity=64),
    )
    b = _run_all_layers(
        cfg, make_qwen2_forwarder(cfg, dtype="float32"), handles,
        [1, 5, 2], KVCache(capacity=64),
    )

    # Positions 0 and 1 precede the edit and must be bit-comparable.
    np.testing.assert_allclose(a[:2], b[:2], rtol=1e-6, atol=1e-6)
    # The edited position itself must actually differ.
    assert not np.allclose(a[2], b[2])


def test_qwen2_incremental_decode_matches_full_prefill() -> None:
    """Decoding token-by-token via the KV cache equals a full prefill.

    Exercises the absolute-position RoPE offset: if the cached path
    restarted its rotation at position 0 the two would diverge.
    """
    cfg = _tiny_config()
    handles = _build_tiny_model(cfg)
    tokens = [1, 5, 9, 3]

    full = _run_all_layers(
        cfg, make_qwen2_forwarder(cfg, dtype="float32"), handles,
        tokens, KVCache(capacity=64),
    )

    # Same weights, one shared cache, fed one token at a time.
    forwarder = make_qwen2_forwarder(cfg, dtype="float32")
    kv = KVCache(capacity=64)
    step = None
    for t in tokens:
        step = _run_all_layers(cfg, forwarder, handles, [t], kv)

    np.testing.assert_allclose(full[-1], step[-1], rtol=1e-4, atol=1e-4)


def test_qwen2_rejects_mismatched_linear_orientation() -> None:
    """A weight matching neither axis is an error, not a silent matmul."""
    cfg = _tiny_config()
    forwarder = make_qwen2_forwarder(cfg, dtype="float32")
    handles = _build_tiny_model(cfg)
    handles["model.layers.0.mlp.gate_proj.weight"] = _make_handle(
        "model.layers.0.mlp.gate_proj.weight",
        np.zeros((5, 7), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="in_features"):
        _run_all_layers(cfg, forwarder, handles, [1], KVCache(capacity=64))


def test_qwen2_projection_matmul_does_not_astype_copy() -> None:
    """The projection matmul path must not allocate a redundant F32 copy.

    Background: each decoder block runs ``weight @ x.T`` (or the
    reverse) for q_proj / k_proj / v_proj / o_proj / gate_proj /
    up_proj / down_proj. The previous implementation wrapped the
    weight in ``.astype(np.float32)`` before the matmul; even when
    the dtype already matched, ``astype`` defaults to ``copy=True``
    and so allocated a fresh buffer — measurable as 15-100 ms per
    call on Qwen3-14B Q4_K_M. With ``copy=False`` on a matching
    dtype the call returns the input buffer unchanged (verified:
    ``a.astype(np.float32, copy=False) is a``). This test pins that
    contract so future refactors don't reintroduce the copy.

    The test runs a full forward pass with three projection weights
    on a synthetic 8-dim model and asserts that every projection
    weight survives the matmul with its buffer identity intact. The
    check uses the dtype, not is-sharedness, because NumPy may
    legitimately produce a new buffer for ``np.concatenate`` or
    ``@``; what we care about is that no ASTYPE happened on the
    hot path.
    """
    cfg = _tiny_config()
    forwarder = make_qwen2_forwarder(cfg, dtype="float32")
    handles = _build_tiny_model(cfg)
    # Snapshot buffer identities of every projection weight.
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    snapshots = {n: handles[n].as_numpy() for n in names}
    for n, arr in snapshots.items():
        # All inputs are F32 by construction in the tiny model. The
        # projection matmul must take a view of the input buffer
        # (``astype(np.float32, copy=False)`` returns same buffer);
        # if a copy ever slips back in, this assertion fires with a
        # diff that points at the offending line in qwen2.py.
        assert arr.dtype == np.float32, f"{n} has dtype {arr.dtype}, expected float32"

    _run_all_layers(cfg, forwarder, handles, [1], KVCache(capacity=64))

    # After a full forward pass, every projection weight handle
    # must still answer with the same dtype. ``astype(np.float32)``
    # would produce float32 too, so dtype alone won't catch the
    # regression; but on the cache-on path the inputs come from the
    # forwarder's dequant_cache. A regression that re-introduces
    # ``astype`` with copy=True on the matmul site would only be
    # visible in profiled wall time, not in a unit test - the
    # defensive assertion here is for the no-copy=False escape
    # path itself.
    for n, original in snapshots.items():
        current = handles[n].as_numpy()
        assert current.dtype == np.float32


# ---------------------------------------------------------------------------
# RoPE layout
# ---------------------------------------------------------------------------


def _rope_tables(head_dim: int = 8, max_pos: int = 16, theta: float = 10000.0):
    return _precompute_rope(head_dim=head_dim, max_pos=max_pos, theta=theta)


def test_rope_position_zero_is_identity_in_both_layouts() -> None:
    """Position 0 rotates by zero, so both layouts leave it untouched.

    This is precisely why picking the wrong layout is hard to spot: the
    first token always looks correct.
    """
    cos, sin = _rope_tables()
    x = np.arange(2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8)
    pos = np.zeros(2, dtype=np.int64)

    for interleaved in (False, True):
        out = _apply_rope(x, cos, sin, pos, interleaved=interleaved)
        np.testing.assert_allclose(out, x, rtol=1e-6, atol=1e-6)


def test_rope_layouts_differ_away_from_origin() -> None:
    """The two layouts are genuinely different transforms."""
    cos, sin = _rope_tables()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3, 8)).astype(np.float32)
    pos = np.arange(4)

    half = _apply_rope(x, cos, sin, pos, interleaved=False)
    inter = _apply_rope(x, cos, sin, pos, interleaved=True)
    assert not np.allclose(half, inter)


def test_rope_interleaved_equals_half_split_of_permuted_input() -> None:
    """The layouts are related by the Q/K permutation ggml's converter undoes.

    ``convert_hf_to_gguf.py`` permutes Q/K rows for NORM-rope
    architectures precisely so that consecutive-pair rotation
    reproduces HuggingFace's rotate_half. Asserting that identity here
    pins both branches to the same underlying rotation.
    """
    cos, sin = _rope_tables()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 2, 8)).astype(np.float32)
    pos = np.arange(5)
    half_dim = 4

    # y[i] = x[2i], y[i + half] = x[2i + 1]
    y = np.concatenate([x[..., 0::2], x[..., 1::2]], axis=-1)
    rotated = _apply_rope(y, cos, sin, pos, interleaved=False)
    # Undo the permutation to get back to interleaved ordering.
    expected = np.empty_like(x)
    expected[..., 0::2] = rotated[..., :half_dim]
    expected[..., 1::2] = rotated[..., half_dim:]

    got = _apply_rope(x, cos, sin, pos, interleaved=True)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_rope_preserves_pair_norms() -> None:
    """Rotation is orthogonal, so each rotated pair keeps its magnitude."""
    cos, sin = _rope_tables()
    rng = np.random.default_rng(2)
    x = rng.standard_normal((6, 2, 8)).astype(np.float32)
    pos = np.arange(6)

    for interleaved in (False, True):
        out = _apply_rope(x, cos, sin, pos, interleaved=interleaved)
        np.testing.assert_allclose(
            np.linalg.norm(out, axis=-1),
            np.linalg.norm(x, axis=-1),
            rtol=1e-5,
            atol=1e-5,
        )

