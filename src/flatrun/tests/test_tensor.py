"""Tests for the TensorHandle and TensorView."""

from __future__ import annotations

import numpy as np
import pytest

from flatrun.core.tensor import (
    BufferTensorHandle,
    HandleSource,
    TensorView,
)
from flatrun.utils.types import TensorKey, TensorMetadata


def _meta(name: str = "x", shape: tuple[int, ...] = (4,), dtype: str = "float32") -> TensorMetadata:
    return TensorMetadata(
        key=TensorKey(file="x.bin", name=name, backend="test"),
        shape=shape,
        dtype=dtype,
        byte_size=int(np.dtype(dtype).itemsize * int(np.prod(shape))),
        offset=0,
    )


def test_buffer_handle_returns_array() -> None:
    meta = _meta(shape=(4,))
    buf = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()
    handle = BufferTensorHandle(meta, buf)
    try:
        view = handle.view()
        assert isinstance(view, TensorView)
        arr = view.as_numpy()
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0, 4.0]))
        # Materialise produces a copy.
        raw = handle.materialize()
        assert raw == buf
    finally:
        handle.close()


def test_close_is_idempotent() -> None:
    meta = _meta()
    handle = BufferTensorHandle(meta, np.zeros(4, dtype=np.float32).tobytes())
    handle.close()
    handle.close()
    assert handle.closed


def test_handle_after_close_raises() -> None:
    meta = _meta()
    handle = BufferTensorHandle(meta, np.zeros(4, dtype=np.float32).tobytes())
    handle.close()
    with pytest.raises(RuntimeError):
        handle.view()


def test_view_copy_returns_independent_array() -> None:
    meta = _meta()
    buf = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32).tobytes()
    handle = BufferTensorHandle(meta, buf)
    try:
        copied = handle.view().as_numpy(copy=True)
        copied[0] = 0.0
        original = handle.view().as_numpy()
        assert original[0] == 10.0
    finally:
        handle.close()


def test_handle_source() -> None:
    meta = _meta()
    handle = BufferTensorHandle(meta, np.zeros(4, dtype=np.float32).tobytes())
    assert handle.source is HandleSource.BUFFER