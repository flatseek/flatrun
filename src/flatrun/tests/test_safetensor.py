"""Tests for the SafeTensor backend."""

from __future__ import annotations

import numpy as np
import pytest

from flatrun.backend.registry import default_registry
from flatrun.backend.safetensor import SafeTensorBackend, open_safetensors
from flatrun.utils.errors import BackendError, TensorNotFoundError


def test_open_lists_tensors(synthetic_model) -> None:
    backend = open_safetensors(synthetic_model)
    try:
        names = {k.name for k in backend.list_tensors()}
        assert "model.embed_tokens.weight" in names
        assert "model.layers.0.proj_0.weight" in names
        assert "model.layers.3.proj_1.weight" in names
        assert "lm_head.weight" in names
    finally:
        backend.close()


def test_metadata_round_trip(synthetic_model) -> None:
    backend = open_safetensors(synthetic_model)
    try:
        meta = backend.get_metadata("model.layers.0.proj_0.weight")
        assert meta.dtype == "float32"
        assert meta.shape == (32, 32)
        assert meta.byte_size == 32 * 32 * 4
        assert meta.offset >= 0
    finally:
        backend.close()


def test_get_missing_tensor_raises(synthetic_model) -> None:
    backend = open_safetensors(synthetic_model)
    try:
        with pytest.raises(TensorNotFoundError):
            backend.get_metadata("does.not.exist")
    finally:
        backend.close()


def test_mmap_handle_returns_zero_copy_view(synthetic_model) -> None:
    backend = open_safetensors(synthetic_model)
    try:
        backend.supports_mmap  # trigger property access
        assert backend.supports_mmap
        handle = backend.open_handle("model.layers.1.proj_0.weight")
        try:
            view = handle.view()
            arr = view.as_numpy()
            assert arr.shape == (32, 32)
            assert arr.dtype == np.float32
            assert not arr.flags.writeable
            # Materialising should give the same bytes.
            raw = handle.materialize()
            assert raw == arr.tobytes()
        finally:
            handle.close()
    finally:
        backend.close()


def test_buffer_handle_when_mmap_disabled(synthetic_model) -> None:
    backend = SafeTensorBackend(synthetic_model, mmap=False)
    backend.open()
    try:
        assert not backend.supports_mmap
        handle = backend.open_handle("model.layers.2.proj_1.weight")
        try:
            assert handle.source.value == "buffer"
            view = handle.view()
            assert view.as_numpy().shape == (32, 32)
        finally:
            handle.close()
    finally:
        backend.close()


def test_double_open_is_idempotent(synthetic_model) -> None:
    backend = SafeTensorBackend(synthetic_model)
    backend.open()
    backend.open()  # second call must not raise
    backend.close()


def test_get_metadata_requires_open(tmp_path) -> None:
    backend = SafeTensorBackend(tmp_path / "missing.safetensors")
    with pytest.raises(BackendError):
        backend.get_metadata("x")


def test_registry_resolves_safetensors(synthetic_model) -> None:
    reg = default_registry()
    backend = reg.open(synthetic_model)
    try:
        assert backend.name == "safetensors"
    finally:
        backend.close()