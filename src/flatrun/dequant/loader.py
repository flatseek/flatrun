"""High-level dequantization helpers.

These wrap the raw dequant functions in convenience APIs that match
the patterns real model files use:

* MLX 4-bit: three tensors per weight, fetched one at a time via the
  runtime's handle cache.
* GGUF K-quants: a single tensor that needs no companion tensors.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.tensor import TensorHandle
from .gguf import (
    dequant_q1_0,
    dequant_q4_0,
    dequant_q4_k,
    dequant_q5_0,
    dequant_q5_1,
    dequant_q5_k,
    dequant_q6_k,
    dequant_q8_0,
)
from .mlx import dequant_mlx_4bit_split


_GGUF_DECODERS = {
    "Q1_0": dequant_q1_0,
    "Q4_0": dequant_q4_0,
    "Q5_0": dequant_q5_0,
    "Q5_1": dequant_q5_1,
    "Q8_0": dequant_q8_0,
    "Q4_K": dequant_q4_k,
    "Q5_K": dequant_q5_k,
    "Q6_K": dequant_q6_k,
}


def dequant_handle(handle: TensorHandle, dtype: str = "float16") -> np.ndarray:
    """Dequantize a single tensor handle.

    The handle's :attr:`metadata.quantization` selects the decoder. When
    the metadata has no quantization tag (``None``) the function falls
    back to ``handle.as_numpy()`` and casts to ``dtype``.
    """
    q = handle.metadata.quantization
    raw = handle.materialize()
    if q is None:
        arr = handle.as_numpy()
        if arr.dtype != np.dtype(dtype):
            arr = arr.astype(dtype)
        return arr
    if q == "MLX_4BIT":
        # MLX handles are split across three tensors; this helper only
        # works for single-tensor formats. Use ``dequant_mlx_weight``
        # for the MLX case.
        raise ValueError("Use dequant_mlx_weight() for MLX 4-bit tensors")
    decoder = _GGUF_DECODERS.get(q)
    if decoder is None:
        raise ValueError(
            f"flatrun has no decoder for ggml quant type {q!r}. "
            f"Supported types: {sorted(_GGUF_DECODERS)}. "
            f"This usually means the GGUF was produced by a fork of "
            f"llama.cpp that added a new quant (Q1_0, IQ3_XXS, ...). "
            f"Re-quantize with `llama-quantize` against the upstream "
            f"build to a supported type (Q4_K_M, Q5_K_M, Q6_K, Q8_0)."
        )
    return decoder(raw, handle.metadata.shape, np.dtype(dtype))


def dequant_mlx_weight(
    acquire: Callable[[str], TensorHandle],
    weight_name: str,
    *,
    dtype: str = "float16",
) -> np.ndarray:
    """Resolve an MLX 4-bit weight (weight + scales + biases) to a NumPy array.

    Parameters
    ----------
    acquire : Callable[[str], TensorHandle]
        Function that resolves a tensor name to a handle. Typically
        :meth:`InferenceRuntime.acquire`.
    weight_name : str
        Base name (e.g. ``"model.layers.0.mlp.gate_proj"``). The
        helper will look up ``{weight_name}.weight``,
        ``{weight_name}.scales``, and ``{weight_name}.biases``.
    dtype : str
        Target dtype (default ``"float16"``).
    """
    w_handle = acquire(f"{weight_name}.weight")
    s_handle = acquire(f"{weight_name}.scales")
    b_handle = acquire(f"{weight_name}.biases")
    # The handles belong to the caller's :class:`LayerHandles` (or the
    # :class:`MemoryManager` cache). Closing them here would invalidate
    # any other layer that subsequently tries to use the same cached
    # handle. The scheduler's ``_release_layer`` is the single owner
    # of the handle lifecycle, so we deliberately don't close them.
    weight = w_handle.as_numpy().astype(np.uint32, copy=False)
    scales = s_handle.as_numpy().astype(np.float16, copy=False)
    biases = b_handle.as_numpy().astype(np.float16, copy=False)
    return dequant_mlx_4bit_split(weight, scales, biases, dtype=dtype)


__all__ = ["dequant_handle", "dequant_mlx_weight"]