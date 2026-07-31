"""Dequantization helpers for non-native dtypes.

FlatRun's storage backends expose tensor bytes through
:class:`TensorHandle`. Native dtypes (F32, F16) come back as ready-to-use
NumPy arrays. Block-quantized dtypes (GGUF Q*_*, MLX 4-bit) need
specialised decoding before they can be used as weights.

This package contains the decoders. They are intentionally pure NumPy
so they run on CPU without any compiled dependencies. Each decoder
takes the raw bytes of one tensor and returns a NumPy array in the
target dtype.

Adding a new decoder:

1. Write a function ``dequant_<name>(raw_bytes, shape, dtype) -> ndarray``
2. Register it in :data:`DECODERS` keyed by the dtype string.
"""

from __future__ import annotations

from typing import Callable, TypeAlias

import numpy as np

from .gguf import dequant_q4_0, dequant_q4_k, dequant_q5_k, dequant_q6_k, dequant_q8_0
from .mlx import dequant_mlx_4bit

DequantFn: TypeAlias = Callable[[bytes, tuple[int, ...], np.dtype], np.ndarray]


DECODERS: dict[str, DequantFn] = {
    "Q4_0": dequant_q4_0,
    "Q8_0": dequant_q8_0,
    "Q4_K": dequant_q4_k,
    "Q5_K": dequant_q5_k,
    "Q6_K": dequant_q6_k,
}


def dequantize(name: str, raw_bytes: bytes, shape: tuple[int, ...], dtype: str = "float16") -> np.ndarray:
    """Dispatch by quant name and return a NumPy array.

    Parameters
    ----------
    name : str
        Quant identifier - one of ``"Q4_0"``, ``"Q8_0"``, ``"Q4_K"``,
        ``"Q5_K"``, ``"Q6_K"``, ``"MLX_4BIT"``.
    raw_bytes : bytes
        Tensor payload exactly as returned by
        :meth:`TensorHandle.materialize`.
    shape : tuple[int, ...]
        Logical shape after dequant.
    dtype : str
        Target dtype - usually ``"float16"`` or ``"float32"``.
    """
    if name == "MLX_4BIT":
        return dequant_mlx_4bit(raw_bytes, shape, dtype)
    fn = DECODERS.get(name)
    if fn is None:
        raise ValueError(f"No dequantizer registered for {name!r}")
    return fn(raw_bytes, shape, np.dtype(dtype))


__all__ = [
    "DECODERS",
    "DequantFn",
    "dequant_mlx_4bit",
    "dequant_q4_0",
    "dequant_q4_k",
    "dequant_q5_k",
    "dequant_q6_k",
    "dequant_q8_0",
    "dequantize",
]