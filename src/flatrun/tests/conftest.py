"""Test fixtures shared across the FlatRun test suite."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def synthetic_model(tmp_path: Path) -> Path:
    """Write a small but realistic SafeTensors file in ``tmp_path``."""
    rng = np.random.default_rng(42)
    layers = 4
    hidden = 32
    tensors: list[tuple[str, tuple[int, ...], str]] = [
        ("model.embed_tokens.weight", (hidden * 4, hidden), "float32"),
        ("model.norm.weight", (hidden,), "float32"),
        ("lm_head.weight", (hidden * 2, hidden), "float32"),
    ]
    for layer_idx in range(layers):
        for proj_idx in range(2):
            tensors.append(
                (f"model.layers.{layer_idx}.proj_{proj_idx}.weight", (hidden, hidden), "float32")
            )

    header: dict[str, object] = {"__metadata__": {"format": "flatrun-tests"}}
    offset = 0
    for name, shape, dtype in tensors:
        size = int(np.dtype(dtype).itemsize * int(np.prod(shape)))
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body = bytearray()
    for _name, shape, dtype in tensors:
        size = int(np.dtype(dtype).itemsize * int(np.prod(shape)))
        body.extend(rng.integers(0, 256, size=size, dtype=np.uint8).tobytes())

    out = tmp_path / "model.safetensors"
    out.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(body))
    return out
