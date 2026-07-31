"""Reference Qwen2 forward pass.

This module implements a CPU-only, pure-NumPy forward pass for the
Qwen2 / Qwen2.5 architecture. It is intentionally simple and slow -
the goal is to demonstrate how a real kernel plugs into FlatRun's
streaming loop, not to compete with hand-optimised GEMM libraries.

Architecture reference: HuggingFace transformers' Qwen2 implementation.
Tensor names follow the HuggingFace convention so the file works with
both SafeTensors and MLX-packaged checkpoints.

Supported variants:

* Dense Qwen2 (any size).
* Grouped-query attention (``num_key_value_heads != num_attention_heads``).
* Tied and untied LM head (controlled by ``tie_word_embeddings``).
* MLX 4-bit quantised weights (each weight stored as ``weight``,
  ``scales``, ``biases`` triples).

What it does NOT do:

* Tokenization - use :mod:`flatrun.tokenizer`.
* Sampling - top-k, top-p, temperature are out of scope.
* Sliding-window attention - the cache grows unbounded.
* Multi-token batches - the attention path is single-position for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..dequant.loader import dequant_handle, dequant_mlx_weight
from ..dequant.mlx import dequant_mlx_4bit_split
from ..runtime.kv_cache import KVCache
from ..runtime.scheduler import LayerHandles
from ..utils.types import LayerDescriptor


def _safe_int(value: object) -> int | None:
    """Coerce a config value to int, returning None when absent/bad."""
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Qwen2Config:
    """Hyperparameters for a Qwen2 / Qwen2.5 model."""

    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    head_dim: int | None = None
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    # Architecture tag. Drives which decoder block the forwarder
    # builds and which Q/K-norm formula it applies. Supported:
    # ``"qwen2"`` (Llama/Qwen2/Qwen2.5/Qwen3), ``"gemma3"``.
    model_arch: str = "qwen2"
    # Pre-attention scale for the attention score. Gemma 3 sometimes
    # exposes this as ``query_pre_attn_scalar`` separate from
    # ``head_dim``; when ``None`` we fall back to ``head_dim``.
    query_pre_attn_scalar: int | None = None
    # When True, per-head Q/K RMSNorm uses ``1 + weight`` as the gain
    # (Gemma 3 convention). When False, plain ``weight`` (Qwen3 / Llama).
    qk_norm_gain: bool = False
    # When True, apply ``post_feedforward_layernorm`` after the MLP
    # residual at the end of each decoder block (Gemma 3).
    mlp_norm_after_block: bool = False
    # Sliding-window attention size. When non-None, each position
    # only attends to the most recent ``sliding_window`` positions
    # (inclusive). Gemma 3 ships with sliding_window=512 on the
    # 1B variant; the larger sizes alternate sliding with full
    # attention per layer, which the current forwarder doesn't
    # model. Setting it on every layer still produces correct
    # outputs on the 1B family.
    sliding_window: int | None = None
    attn_logit_softcap: float | None = None
    # RoPE pair layout. False = HuggingFace ``rotate_half`` (ggml calls
    # it NEOX); True = consecutive pairs (ggml NORM). HF-layout
    # checkpoints are always False. GGUF files depend on architecture:
    # ``convert_hf_to_gguf.py`` un-permutes Q/K for the NORM arches
    # (llama, baichuan, starcoder, ...) but leaves the NEOX ones
    # (qwen2, phi3, gemma, ...) alone. See :func:`_apply_rope`.
    rope_interleaved: bool = False
    # Quantisation strategy. At most one of these should be set.
    quant_mlx_4bit: bool = False   # MLX 4-bit (weight + scales + biases triples).
    quant_gguf: str | None = None   # GGUF quant name like "Q8_0", "Q4_K", "Q6_K".

    @classmethod
    def from_hf_config(cls, raw: dict[str, object]) -> "Qwen2Config":
        """Build a config from a parsed ``config.json``.

        Architecture detection looks at ``architectures`` and
        ``model_type``. Anything that isn't explicitly Gemma gets the
        Qwen2 / Llama path - that's the broader family Qwen2.5,
        Qwen3, Llama 1-3 and SmolLM2 all live in.
        """
        architectures = raw.get("architectures") or ()
        model_type = str(raw.get("model_type") or "")
        arch = "gemma3" if (
            (architectures and any("Gemma3" in str(a) for a in architectures))
            or "gemma3" in model_type
        ) else "qwen2"

        tie = raw.get("tie_word_embeddings")
        if tie is None:
            # Most Gemma 3 sizes ship with tied embeddings; Llama /
            # Qwen default to untied. The HuggingFace convention is
            # that a missing key is *not* a hard "false" - it just
            # means "trust the architecture defaults".
            tie = arch != "gemma3"

        return cls(
            vocab_size=int(raw.get("vocab_size", 152064)),
            hidden_size=int(raw.get("hidden_size", 3584)),
            intermediate_size=int(raw.get("intermediate_size", 18944)),
            num_hidden_layers=int(raw.get("num_hidden_layers", 28)),
            num_attention_heads=int(raw.get("num_attention_heads", 28)),
            num_key_value_heads=int(raw.get("num_key_value_heads", 4)),
            head_dim=_safe_int(raw.get("head_dim")),
            max_position_embeddings=int(raw.get("max_position_embeddings", 32768)),
            rope_theta=float(raw.get("rope_theta", 1000000.0)),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            tie_word_embeddings=bool(tie),
            # HF checkpoints are always stored in rotate_half layout.
            # ``rope_interleaved`` in an HF config.json refers to the
            # same distinction, so honour it when present.
            rope_interleaved=bool(raw.get("rope_interleaved", False)),
            model_arch=arch,
            # Gemma 3 sometimes uses ``query_pre_attn_scalar`` as a
            # separate value from ``head_dim`` for the attention
            # score scale. When absent, ``head_dim`` is the fallback.
            query_pre_attn_scalar=_safe_int(raw.get("query_pre_attn_scalar")),
            qk_norm_gain=arch == "gemma3",
            mlp_norm_after_block=arch == "gemma3",
            sliding_window=_safe_int(raw.get("sliding_window")),
            attn_logit_softcap=_safe_int(raw.get("attn_logit_softcap"))
            or (50.0 if arch == "gemma3" else None),
        )


# ---------------------------------------------------------------------------
# Building blocks (pure NumPy)
# ---------------------------------------------------------------------------


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Apply RMSNorm: ``x * rsqrt(mean(x^2) + eps) * weight``.

    The whole computation stays in float32. Rounding the normalised
    activations back to the input dtype *before* applying the gain (as
    an earlier version did) throws away ~3 decimal digits on every one
    of the 2N norms in the stack, and the error compounds through the
    residual path.
    """
    x32 = x.astype(np.float32, copy=False)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    return x32 * inv * weight.astype(np.float32, copy=False)


def _gemma3_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Gemma 3 RMSNorm variant.

    Same normalisation as :func:`_rms_norm` but the gain is
    ``1 + weight`` rather than ``weight``. The HF initialisation sets
    ``weight`` to zero, so the two formulations are equivalent at
    initialisation; the saved checkpoint has trained the gain away
    from zero and switching them quietly doubles or halves the
    Q/K activation magnitude, which is enough to flatten the
    attention map and turn the model into a punctuation sampler.
    """
    x32 = x.astype(np.float32, copy=False)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    return x32 * inv * (1.0 + weight.astype(np.float32, copy=False))


def _precompute_rope(
    head_dim: int,
    max_pos: int,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute cos/sin tables for rotary embeddings.

    Returns (cos, sin) of shape ``(max_pos, head_dim // 2)`` in float32 -
    one angle per rotated *pair*, which is all either RoPE layout needs.
    Keeping the tables in float32 matters: in float16 the angles for
    positions past ~2048 quantise badly enough to visibly rotate Q/K
    away from where the model was trained to expect them.
    """
    inv_freq = 1.0 / (
        theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    positions = np.arange(max_pos, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)  # (max_pos, head_dim // 2)
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def _apply_rope(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    positions: np.ndarray,
    *,
    interleaved: bool,
) -> np.ndarray:
    """Apply rotary embeddings to a whole ``(seq, heads, head_dim)`` block.

    ``positions`` gives the absolute position of each row of ``x`` -
    absolute, not relative, so a decode step continuing a cached prefix
    rotates by the true index rather than restarting at zero.

    Two layouts exist and they are **not** interchangeable:

    * ``interleaved=False`` ("half-split", what ggml calls NEOX) pairs
      dimension ``i`` with ``i + head_dim/2``. This is HuggingFace's
      ``rotate_half`` and applies to every checkpoint stored in HF
      layout, plus GGUF files whose architecture ggml maps to NEOX
      (``qwen2``, ``phi3``, ``gemma``, ``stablelm``, ...).
    * ``interleaved=True`` (what ggml calls NORM) pairs *consecutive*
      dimensions ``2i`` and ``2i+1``. ``convert_hf_to_gguf.py``
      un-permutes Q/K for these architectures, so a GGUF whose arch
      ggml maps to NORM (``llama``, ``baichuan``, ``starcoder``, ...)
      needs this one.

    Picking the wrong layout leaves position 0 exactly right - its
    rotation is the identity either way - and quietly corrupts every
    position after it, which is why the damage shows up as plausible
    but wrong continuations rather than an obvious crash.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    x32 = x.astype(np.float32, copy=False)
    # (seq, 1, half) so it broadcasts across the head axis.
    cos_p = cos[positions][:, None, :]
    sin_p = sin[positions][:, None, :]

    if interleaved:
        pairs = x32.reshape(*x32.shape[:-1], half, 2)
        even = pairs[..., 0]
        odd = pairs[..., 1]
        rot = np.stack(
            [even * cos_p - odd * sin_p, odd * cos_p + even * sin_p],
            axis=-1,
        )
        return rot.reshape(x32.shape)

    x_first = x32[..., :half]
    x_second = x32[..., half:]
    return np.concatenate(
        [
            x_first * cos_p - x_second * sin_p,
            x_second * cos_p + x_first * sin_p,
        ],
        axis=-1,
    )


def _causal_mask(seq_len: int, past_len: int, sliding_window: int | None = None) -> np.ndarray:
    """Additive attention mask of shape ``(seq_len, past_len + seq_len)``.

    Query row ``t`` may attend to key column ``j`` only when
    ``j <= past_len + t``. Without this every prefill position sees the
    whole future of the prompt, which quietly corrupts the hidden
    states that later positions then attend to - the single largest
    source of fluent-looking nonsense in a hand-written decoder.

    ``sliding_window`` further restricts each row to the most recent
    ``sliding_window`` keys. Gemma 3 ships with ``sliding_window=512``;
    the 1B variant applies it to every layer, so missing it produces
    a flat, placeholder-laden distribution rather than the
    logit-diverse output of a correctly-windowed model.
    """
    total = past_len + seq_len
    q_pos = np.arange(seq_len, dtype=np.int32)[:, None] + past_len
    k_pos = np.arange(total, dtype=np.int32)[None, :]
    mask = np.zeros((seq_len, total), dtype=np.float32)
    mask[k_pos > q_pos] = -np.inf
    if sliding_window is not None and sliding_window > 0:
        # Position ``t`` may only see keys in
        # ``[t - sliding_window + 1, t]``.
        mask[k_pos < q_pos - sliding_window + 1] = -np.inf
    return mask


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)



# ---------------------------------------------------------------------------
# Tensor fetch helpers
# ---------------------------------------------------------------------------


def _as_linear(weight: np.ndarray, in_features: int, name: str) -> np.ndarray:
    """Normalise a linear weight to the canonical ``(out, in)`` layout.

    SafeTensors (``nn.Linear.weight``), GGUF (``ne[0]`` is the input
    dim, and numpy sees the dims reversed) and MLX all store linear
    weights as ``(out_features, in_features)``. This helper asserts
    that and repairs the transposed case when the shape makes the
    orientation unambiguous, so a mislabelled backend fails loudly
    instead of silently multiplying by a transposed matrix - which is
    shape-legal for every square projection and produces exactly the
    kind of fluent garbage that is hard to trace back.
    """
    if weight.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D linear weight, got {weight.shape}")
    if weight.shape[1] == in_features:
        return weight
    if weight.shape[0] == in_features:
        return weight.T
    raise ValueError(
        f"{name}: neither axis of {weight.shape} matches in_features={in_features}"
    )


def _fetch_proj(
    handles: LayerHandles,
    layer_index: int,
    proj_name: str,
    config: Qwen2Config,
    dtype: np.dtype,
    *,
    dequant_cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Resolve a projection weight to a NumPy array, with dequant.

    ``proj_name`` is the *short* name within the layer (e.g.
    ``"self_attn.q_proj"``, ``"mlp.gate_proj"``). The function looks
    up ``model.layers.<i>.<proj_name>.weight`` (and ``.scales`` /
    ``.biases`` for MLX).

    The result is always in the on-disk ``(out_features, in_features)``
    layout; callers multiply with ``x @ W.T``. Use :func:`_as_linear`
    to check that assumption where the input dim is known.

    If ``dequant_cache`` is supplied, the dequantized result is
    memoised under ``f"{base}.weight"`` so repeated calls across
    steps return the same buffer. The cache lives on the forwarder
    closure; evict it (e.g. ``cache.clear()``) when the underlying
    weights may have changed.
    """
    if layer_index == -1:
        # Pre/post-layer tensor (embedding, LM head, etc.).
        base = proj_name
    else:
        base = f"model.layers.{layer_index}.{proj_name}"
    cache_key = f"{base}.weight"
    if config.quant_mlx_4bit:
        if dequant_cache is not None and cache_key in dequant_cache:
            return dequant_cache[cache_key]
        arr = dequant_mlx_weight(
            lambda n: handles[n],
            base,
            dtype=dtype.name,
        )
        if dequant_cache is not None:
            dequant_cache[cache_key] = arr
        return arr
    weight_handle = handles[f"{base}.weight"]
    if weight_handle.metadata.quantization is not None:
        if dequant_cache is not None and cache_key in dequant_cache:
            return dequant_cache[cache_key]
        arr = dequant_handle(weight_handle, dtype=dtype.name)
        if dequant_cache is not None:
            dequant_cache[cache_key] = arr
    else:
        arr = weight_handle.as_numpy().astype(dtype)
    return arr


def _fetch_linear_with_quant(
    handles: LayerHandles,
    layer_index: int,
    proj_name: str,
    config: Qwen2Config,
    dtype: np.dtype,
    *,
    dequant_cache: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fetch a linear weight and its bias (if present).

    Returns ``(weight, bias_or_none)``. MLX dequant handled inside
    ``_fetch_proj``.
    """
    weight = _fetch_proj(
        handles,
        layer_index,
        proj_name,
        config,
        dtype,
        dequant_cache=dequant_cache,
    )
    if layer_index == -1:
        bias_name = f"{proj_name}.bias"
    else:
        bias_name = f"model.layers.{layer_index}.{proj_name}.bias"
    if bias_name in handles:
        bias = handles[bias_name].as_numpy().astype(np.float32)
    else:
        bias = None
    return weight, bias


# ---------------------------------------------------------------------------
# The forward pass
# ---------------------------------------------------------------------------


def make_qwen2_forwarder(
    config: Qwen2Config,
    *,
    dtype: str = "float32",
) -> Callable[[LayerDescriptor, LayerHandles, KVCache], np.ndarray]:
    """Return a ``ForwardFn`` that runs a Qwen2 / Llama forward pass.

    The returned callable accepts ``(layer, handles, kv_cache)`` and

    * on layer 0 embeds the token ids and then runs decoder block 0,
    * on the last layer runs the decoder block **and then** the final
      norm + LM head, returning ``(seq, vocab)`` logits,
    * on every other layer runs the decoder block and returns hidden
      states.

    Activations are computed in float32 regardless of the storage
    dtype. ``dtype`` only controls how weights are materialised.
    """
    np_dtype = np.dtype(dtype)
    head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    if n_heads % n_kv_heads:
        raise ValueError(
            f"num_attention_heads={n_heads} is not a multiple of "
            f"num_key_value_heads={n_kv_heads}"
        )
    head_group = n_heads // n_kv_heads  # GQA group size
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    last_index = config.num_hidden_layers - 1

    # Per-forwarder dequant cache, keyed on the on-disk tensor name so
    # different layers keep separate entries. Without it every step
    # re-dequantises every weight it touches.
    dequant_cache: dict[str, np.ndarray] = {}

    cos_cache, sin_cache = _precompute_rope(
        head_dim=head_dim,
        max_pos=config.max_position_embeddings,
        theta=config.rope_theta,
    )

    state: dict[str, np.ndarray | None] = {"hidden": None, "embed": None}

    def _embedding(handles: LayerHandles) -> np.ndarray:
        """Fetch the token embedding table as ``(vocab, hidden)``."""
        embed = _fetch_proj(
            handles, -1, "model.embed_tokens", config, np_dtype,
            dequant_cache=dequant_cache,
        )
        if embed.ndim != 2:
            raise ValueError(f"embed_tokens must be 2-D, got {embed.shape}")
        # Unambiguous: the vocab axis is never the hidden axis in any
        # model we support, so orient by hidden_size rather than by
        # trusting a per-format convention.
        if embed.shape[1] != config.hidden_size:
            if embed.shape[0] == config.hidden_size:
                embed = embed.T
            else:
                raise ValueError(
                    f"embed_tokens {embed.shape} matches neither axis of "
                    f"hidden_size={config.hidden_size}"
                )
        return embed

    def _decoder_block(
        idx: int,
        handles: LayerHandles,
        hidden: np.ndarray,
        kv: KVCache,
    ) -> np.ndarray:
        """One decoder block: attn (RoPE + causal) -> MLP, both residual."""
        seq_len = hidden.shape[0]

        # ----- Attention -----
        attn_norm_w = handles[
            f"model.layers.{idx}.input_layernorm.weight"
        ].as_numpy().astype(np.float32)
        residual = hidden
        x = _rms_norm(hidden, attn_norm_w, config.rms_norm_eps)

        q_w, q_b = _fetch_linear_with_quant(
            handles, idx, "self_attn.q_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        k_w, k_b = _fetch_linear_with_quant(
            handles, idx, "self_attn.k_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        v_w, v_b = _fetch_linear_with_quant(
            handles, idx, "self_attn.v_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        # Canonical (out, in); one matmul for all three projections.
        q_w = _as_linear(q_w, config.hidden_size, f"layer{idx}.q_proj")
        k_w = _as_linear(k_w, config.hidden_size, f"layer{idx}.k_proj")
        v_w = _as_linear(v_w, config.hidden_size, f"layer{idx}.v_proj")
        qkv_w = np.concatenate([q_w, k_w, v_w], axis=0).astype(np.float32)
        qkv = x @ qkv_w.T
        if q_b is not None or k_b is not None or v_b is not None:
            qkv = qkv + np.concatenate(
                [
                    q_b if q_b is not None else np.zeros(q_w.shape[0], np.float32),
                    k_b if k_b is not None else np.zeros(k_w.shape[0], np.float32),
                    v_b if v_b is not None else np.zeros(v_w.shape[0], np.float32),
                ],
                axis=0,
            )

        q = qkv[:, :q_dim].reshape(seq_len, n_heads, head_dim)
        k = qkv[:, q_dim : q_dim + kv_dim].reshape(seq_len, n_kv_heads, head_dim)
        v = qkv[:, q_dim + kv_dim :].reshape(seq_len, n_kv_heads, head_dim)

        # Rotate at the *absolute* position, which continues any cached
        # prefix instead of restarting at zero.
        past = kv.stack(idx)
        past_len = 0 if past is None else int(past[0].shape[0])
        positions = np.arange(past_len, past_len + seq_len)

        # Qwen3 applies per-head RMSNorm to the head axis of Q and K
        # *before* RoPE, using ``attn_q_norm`` / ``attn_k_norm`` of
        # length ``head_dim``. Qwen2 / Llama have no such tensor - the
        # lookup is a no-op there. Skipping the norm on Qwen3 leaves
        # the activations at the wrong scale and the model collapses
        # to a low-rank output that reads as a stream of punctuation.
        # GGUF names these ``attn_q_norm`` / ``attn_k_norm`` (no
        # ``self_attn.`` prefix); HF's Qwen3 forward uses the latter.
        # The check handles either convention so a per-layer look-up
        # works regardless of how the checkpoint was serialised.
        q_norm_w = (
            handles.get(f"model.layers.{idx}.self_attn.q_norm.weight")
            or handles.get(f"model.layers.{idx}.attn_q_norm.weight")
        )
        k_norm_w = (
            handles.get(f"model.layers.{idx}.self_attn.k_norm.weight")
            or handles.get(f"model.layers.{idx}.attn_k_norm.weight")
        )
        if q_norm_w is not None or k_norm_w is not None:
            if q_norm_w is None or k_norm_w is None:
                raise ValueError(
                    f"layer {idx}: q_norm/k_norm must come as a pair; "
                    f"got q={q_norm_w is not None} k={k_norm_w is not None}"
                )
            q = _rms_norm(q, q_norm_w.as_numpy().astype(np.float32), config.rms_norm_eps)
            k = _rms_norm(k, k_norm_w.as_numpy().astype(np.float32), config.rms_norm_eps)

        q = _apply_rope(q, cos_cache, sin_cache, positions,
                        interleaved=config.rope_interleaved)
        k = _apply_rope(k, cos_cache, sin_cache, positions,
                        interleaved=config.rope_interleaved)

        for pos in range(seq_len):
            kv.append(idx, k[pos], v[pos])
        k_hist, v_hist = kv.stack(idx)

        # GQA: replicate each KV head across its query group. repeat (not
        # tile) is what pairs head h with kv head h // head_group.
        k_full = np.repeat(k_hist, head_group, axis=1)
        v_full = np.repeat(v_hist, head_group, axis=1)

        scale = 1.0 / np.sqrt(head_dim)
        attn = np.einsum("thd,Thd->htT", q, k_full) * scale
        attn = attn + _causal_mask(seq_len, past_len)
        if config.attn_logit_softcap is not None:
            cap = float(config.attn_logit_softcap)
            attn = np.tanh(attn / cap) * cap
        attn = _softmax(attn, axis=-1)
        context = np.einsum("htT,Thd->thd", attn, v_full)
        attn_out = context.reshape(seq_len, q_dim)

        o_w, o_b = _fetch_linear_with_quant(
            handles, idx, "self_attn.o_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        o_w = _as_linear(o_w, q_dim, f"layer{idx}.o_proj")
        attn_out = attn_out @ o_w.astype(np.float32).T
        if o_b is not None:
            attn_out = attn_out + o_b
        hidden = residual + attn_out

        # ----- MLP (SwiGLU) -----
        mlp_norm_w = handles[
            f"model.layers.{idx}.post_attention_layernorm.weight"
        ].as_numpy().astype(np.float32)
        residual = hidden
        x = _rms_norm(hidden, mlp_norm_w, config.rms_norm_eps)

        gate_w, _ = _fetch_linear_with_quant(
            handles, idx, "mlp.gate_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        up_w, _ = _fetch_linear_with_quant(
            handles, idx, "mlp.up_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        down_w, _ = _fetch_linear_with_quant(
            handles, idx, "mlp.down_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        gate_w = _as_linear(gate_w, config.hidden_size, f"layer{idx}.gate_proj")
        up_w = _as_linear(up_w, config.hidden_size, f"layer{idx}.up_proj")
        inter = gate_w.shape[0]
        down_w = _as_linear(down_w, inter, f"layer{idx}.down_proj")

        gateup = x @ np.concatenate([gate_w, up_w], axis=0).astype(np.float32).T
        gate = gateup[:, :inter]
        up = gateup[:, inter:]
        # SiLU(gate) * up. Clip guards exp() from overflowing to inf on
        # the rare large-magnitude activation.
        silu_gate = gate / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0)))
        mlp_out = (silu_gate * up) @ down_w.astype(np.float32).T

        return residual + mlp_out

    def _gemma3_decoder_block(
        idx: int,
        handles: LayerHandles,
        hidden: np.ndarray,
        kv: KVCache,
    ) -> np.ndarray:
        """One Gemma 3 decoder block.

        Differs from the Qwen2 / Llama path in three ways:

        * Every RMSNorm in the block - ``input_layernorm``,
          ``pre_feedforward_layernorm``, ``post_feedforward_layernorm``,
          and the per-head Q / K norms - uses the Gemma 3 convention
          ``x * rsqrt(...) * (1 + weight)`` rather than the plain
          ``x * rsqrt(...) * weight``. The weights are initialised to
          zero, so the two formulations match at initialisation; a
          trained checkpoint has shifted the gain away from zero, so
          substituting plain RMSNorm quietly doubles or halves the
          activation magnitude. See :func:`_gemma3_norm`.
        * The attention score is scaled by
          ``sqrt(query_pre_attn_scalar)`` rather than
          ``sqrt(head_dim)``. Gemma 3 sometimes exposes these as the
          same value but they don't have to be.
        * The MLP block is bounded by *two* RMSNorms:
          ``pre_feedforward_layernorm`` before the matmuls and
          ``post_feedforward_layernorm`` after the MLP residual. The
          older ``post_attention_layernorm`` is not referenced by the
          decoder block in HF ``Gemma3TextDecoderLayer``.
        """
        seq_len = hidden.shape[0]
        attn_scale = config.query_pre_attn_scalar or head_dim

        # ----- Attention -----
        attn_norm_w = handles[
            f"model.layers.{idx}.input_layernorm.weight"
        ].as_numpy().astype(np.float32)
        residual = hidden
        x = _gemma3_norm(hidden, attn_norm_w, config.rms_norm_eps)

        q_w, _ = _fetch_linear_with_quant(
            handles, idx, "self_attn.q_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        k_w, _ = _fetch_linear_with_quant(
            handles, idx, "self_attn.k_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        v_w, _ = _fetch_linear_with_quant(
            handles, idx, "self_attn.v_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        q_w = _as_linear(q_w, config.hidden_size, f"layer{idx}.q_proj")
        k_w = _as_linear(k_w, config.hidden_size, f"layer{idx}.k_proj")
        v_w = _as_linear(v_w, config.hidden_size, f"layer{idx}.v_proj")
        qkv_w = np.concatenate([q_w, k_w, v_w], axis=0).astype(np.float32)
        qkv = x @ qkv_w.T

        q = qkv[:, :q_dim].reshape(seq_len, n_heads, head_dim)
        k = qkv[:, q_dim : q_dim + kv_dim].reshape(seq_len, n_kv_heads, head_dim)
        v = qkv[:, q_dim + kv_dim :].reshape(seq_len, n_kv_heads, head_dim)

        past = kv.stack(idx)
        past_len = 0 if past is None else int(past[0].shape[0])
        positions = np.arange(past_len, past_len + seq_len)

        # Q / K RMSNorm with Gemma's ``1 + weight`` gain. The norm
        # is applied to each (seq, head, head_dim) slice, *before*
        # RoPE - same placement as Qwen3.
        q_norm_w = handles[
            f"model.layers.{idx}.self_attn.q_norm.weight"
        ].as_numpy().astype(np.float32)
        k_norm_w = handles[
            f"model.layers.{idx}.self_attn.k_norm.weight"
        ].as_numpy().astype(np.float32)
        q = _gemma3_norm(q, q_norm_w, config.rms_norm_eps)
        k = _gemma3_norm(k, k_norm_w, config.rms_norm_eps)

        q = _apply_rope(q, cos_cache, sin_cache, positions,
                        interleaved=config.rope_interleaved)
        k = _apply_rope(k, cos_cache, sin_cache, positions,
                        interleaved=config.rope_interleaved)

        for pos in range(seq_len):
            kv.append(idx, k[pos], v[pos])
        k_hist, v_hist = kv.stack(idx)

        k_full = np.repeat(k_hist, head_group, axis=1)
        v_full = np.repeat(v_hist, head_group, axis=1)

        scale = 1.0 / np.sqrt(attn_scale)
        attn = np.einsum("thd,Thd->htT", q, k_full) * scale
        attn = attn + _causal_mask(seq_len, past_len, config.sliding_window)
        # Gemma 2/3 default soft-cap (50.0) prevents the Q.K product
        # from saturating the softmax once the trained RMSNorm gain
        # pushes activations past ~30x at late layers. Without this
        # every position attends to a single key and the residual
        # stream collapses across positions, producing flat logits.
        if config.attn_logit_softcap is not None:
            cap = float(config.attn_logit_softcap)
            attn = np.tanh(attn / cap) * cap
        attn = _softmax(attn, axis=-1)
        context = np.einsum("htT,Thd->thd", attn, v_full)
        attn_out = context.reshape(seq_len, q_dim)

        o_w, _ = _fetch_linear_with_quant(
            handles, idx, "self_attn.o_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        o_w = _as_linear(o_w, q_dim, f"layer{idx}.o_proj")
        attn_out = attn_out @ o_w.astype(np.float32).T
        hidden = residual + attn_out

        # ----- MLP (SwiGLU) with pre/post norms -----
        pre_mlp_norm_w = handles[
            f"model.layers.{idx}.pre_feedforward_layernorm.weight"
        ].as_numpy().astype(np.float32)
        residual = hidden
        x = _gemma3_norm(hidden, pre_mlp_norm_w, config.rms_norm_eps)

        gate_w, _ = _fetch_linear_with_quant(
            handles, idx, "mlp.gate_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        up_w, _ = _fetch_linear_with_quant(
            handles, idx, "mlp.up_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        down_w, _ = _fetch_linear_with_quant(
            handles, idx, "mlp.down_proj", config, np_dtype, dequant_cache=dequant_cache
        )
        gate_w = _as_linear(gate_w, config.hidden_size, f"layer{idx}.gate_proj")
        up_w = _as_linear(up_w, config.hidden_size, f"layer{idx}.up_proj")
        inter = gate_w.shape[0]
        down_w = _as_linear(down_w, inter, f"layer{idx}.down_proj")

        gateup = x @ np.concatenate([gate_w, up_w], axis=0).astype(np.float32).T
        gate = gateup[:, :inter]
        up = gateup[:, inter:]
        silu_gate = gate / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0)))
        mlp_out = (silu_gate * up) @ down_w.astype(np.float32).T
        hidden = residual + mlp_out

        # Gemma 3 applies ``post_feedforward_layernorm`` to the *whole*
        # block's output (after the MLP residual) before handing it to
        # the next layer. This is what differentiates Gemma's residual
        # pattern from Qwen2 / Llama.
        post_mlp_norm_w = handles[
            f"model.layers.{idx}.post_feedforward_layernorm.weight"
        ].as_numpy().astype(np.float32)
        hidden = _gemma3_norm(hidden, post_mlp_norm_w, config.rms_norm_eps)
        return hidden

    def forward(
        layer: LayerDescriptor,
        handles: LayerHandles,
        kv: KVCache,
    ) -> np.ndarray:
        idx = layer.index

        if idx == 0:
            tokens = np.asarray(handles.tokens or [], dtype=np.int64)
            if tokens.size == 0:
                raise ValueError("forward() called with no tokens bound")
            embed = _embedding(handles)
            # Stash it so a tied LM head can reuse the same buffer
            # instead of pinning the source tensor for the whole pass.
            state["embed"] = embed
            state["hidden"] = embed[tokens].astype(np.float32)

        hidden = state["hidden"]
        assert hidden is not None
        hidden = (
            _gemma3_decoder_block(idx, handles, hidden, kv)
            if config.model_arch == "gemma3"
            else _decoder_block(idx, handles, hidden, kv)
        )
        state["hidden"] = hidden

        if idx != last_index:
            return hidden

        # ----- Final norm + LM head, after the last block has run -----
        norm_w = handles["model.norm.weight"].as_numpy().astype(np.float32)
        # Gemma 3's ``model.norm`` is also a Gemma3RMSNorm
        # (``1 + weight`` gain); every other arch uses plain RMSNorm.
        if config.model_arch == "gemma3":
            hidden = _gemma3_norm(hidden, norm_w, config.rms_norm_eps)
        else:
            hidden = _rms_norm(hidden, norm_w, config.rms_norm_eps)
        if config.tie_word_embeddings:
            head_w = state["embed"]
            assert head_w is not None
        else:
            head_w = _fetch_proj(
                handles, -1, "lm_head", config, np_dtype,
                dequant_cache=dequant_cache,
            )
            head_w = _as_linear(head_w, config.hidden_size, "lm_head")
        return (hidden @ head_w.astype(np.float32).T).astype(np.float32)

    return forward


__all__ = ["Qwen2Config", "make_qwen2_forwarder"]