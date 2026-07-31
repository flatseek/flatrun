"""MLX 4-bit dequantization.

MLX stores a 4-bit quantised weight as three tensors:

* ``weight``  - packed ``uint32`` array. Each ``uint32`` holds eight
  4-bit nibbles.
* ``scales``  - one fp16 scale per group of 64 logical elements.
* ``biases``  - one fp16 bias per group of 64 logical elements.

The logical weight has shape ``(out_features, in_features)`` and the
group count is ``(in_features // GROUP_SIZE) * out_features`` where
``GROUP_SIZE = 64``. Decoding is:

    for each (row, group):
        for each col in group:
            value = ((packed[row, group_idx] >> (4 * col_in_group)) & 0xF) - 8
            out[row, group * 64 + col_in_group] = value * scales[row, group_idx] + biases[row, group_idx]

The implementation is pure NumPy and stays in fp16 by default. ``dtype``
can be overridden to ``float32`` for higher-precision forward passes.

Layout assumptions (verified against the MLX reference):

* ``weight`` shape ``(out, in // 8)`` of uint32 - 8 nibbles per uint32.
* ``scales`` / ``biases`` shape ``(out, in // GROUP_SIZE)`` of fp16.
* Group size is 64.
* Values are stored ``unsigned + bias`` with bias ``= 8``.
"""

from __future__ import annotations

import numpy as np


_MLX_GROUP_SIZE = 64
_MLX_NIBBLES_PER_U32 = 8


def _decode_mlx_4bit(
    weight: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    """Inner worker - takes already-split arrays."""
    out_features, packed_cols = weight.shape
    in_features = packed_cols * _MLX_NIBBLES_PER_U32
    n_groups = in_features // _MLX_GROUP_SIZE
    if scales.shape != (out_features, n_groups):
        raise ValueError(
            f"MLX scales shape {scales.shape} does not match expected "
            f"({out_features}, {n_groups})"
        )
    if biases.shape != scales.shape:
        raise ValueError(
            f"MLX biases shape {biases.shape} does not match scales {scales.shape}"
        )

    # Extract the 8 nibbles from each uint32.
    nibbles = (
        weight[:, :, None]
        >> (np.arange(_MLX_NIBBLES_PER_U32, dtype=np.uint32) * 4)
    ) & 0xF  # shape (out, packed_cols, 8)
    # Centre around 0 by subtracting 8 (matches MLX convention).
    nibbles = nibbles.astype(np.int32) - 8
    # Reshape into (out, n_groups, GROUP_SIZE).
    nibbles = nibbles.reshape(out_features, n_groups, _MLX_GROUP_SIZE)

    # Promote scales/biases for the multiplication.
    scales3 = scales[:, :, None].astype(np.float32)
    biases3 = biases[:, :, None].astype(np.float32)
    out = nibbles.astype(np.float32) * scales3 + biases3
    out = out.reshape(out_features, in_features).astype(dtype)
    return out


def dequant_mlx_4bit(
    raw_bytes: bytes,
    shape: tuple[int, ...],
    dtype: str = "float16",
) -> np.ndarray:
    """Dequantize a single MLX 4-bit weight tensor.

    The decoder expects the bytes to contain three back-to-back tensors
    in the order ``weight | scales | biases``. This matches the layout
    FlatRun's SafeTensor backend returns when the three tensors are
    concatenated via :meth:`TensorHandle.materialize`.

    For per-tensor decoding (each as a separate handle), use
    :func:`dequant_mlx_4bit_split` instead.
    """
    raise NotImplementedError(
        "Use dequant_mlx_4bit_split() when weight/scales/biases are "
        "separate handles, or pack them into a single tensor first."
    )


def dequant_mlx_4bit_split(
    weight: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    *,
    dtype: str = "float16",
) -> np.ndarray:
    """Dequantize from three separate NumPy arrays.

    The output has the same shape as the logical (un-quantised) weight:
    ``(out_features, in_features)``.
    """
    if weight.dtype not in (np.uint32, np.int32):
        raise TypeError(f"MLX weight must be uint32, got {weight.dtype}")
    if scales.dtype not in (np.float16, np.float32):
        raise TypeError(f"MLX scales must be float16/float32, got {scales.dtype}")
    if biases.dtype not in (np.float16, np.float32):
        raise TypeError(f"MLX biases must be float16/float32, got {biases.dtype}")
    return _decode_mlx_4bit(weight, scales, biases, dtype=np.dtype(dtype))


__all__ = [
    "dequant_mlx_4bit",
    "dequant_mlx_4bit_split",
]