"""Tests for the Backend facade dispatcher.

Covers:
- BackendBase.dispatch() routes to the right per-quant method
- PythonBackend handles every supported quant via numpy dequant
- NativeBackend falls back to Python when the C++ extension is unavailable
- Unknown quant names fall back to F32 @ x
- get_backend() rejects unknown names
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from flatrun.runtime.backend import (  # noqa: E402
    BackendBase,
    NativeBackend,
    PythonBackend,
    _QUANT_DISPATCH,
    _quant_method,
    get_backend,
)


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------


def test_quant_method_dispatch_table() -> None:
    """Every quant in _QUANT_DISPATCH resolves to a method name."""
    for quant, method in _QUANT_DISPATCH.items():
        assert _quant_method(quant) == method
    # Unknown quant returns None
    assert _quant_method("F32") is None
    assert _quant_method(None) is None
    assert _quant_method("UNKNOWN") is None


def test_python_backend_name() -> None:
    py = PythonBackend()
    assert py.name == "python"
    assert py.available is True


def test_native_backend_name() -> None:
    n = NativeBackend()
    assert n.name == "native"
    # available is bool regardless of whether _C is built
    assert isinstance(n.available, bool)


# ---------------------------------------------------------------------------
# Supported quants
# ---------------------------------------------------------------------------


def test_python_backend_supported_quants() -> None:
    py = PythonBackend()
    quants = py.supported_quants
    assert "Q4_K" in quants
    assert "Q6_K" in quants
    assert "Q8_0" in quants
    # Python backend also handles non-quant formats
    assert "F32" in quants
    assert "F16" in quants


def test_native_backend_supported_quants() -> None:
    n = NativeBackend()
    if n.available:
        assert n.supported_quants == {"Q4_K", "Q6_K", "Q8_0"}
    else:
        # No C++ extension built — empty support set
        assert n.supported_quants == set()


# ---------------------------------------------------------------------------
# get_backend
# ---------------------------------------------------------------------------


def test_get_backend_python() -> None:
    b = get_backend("python")
    assert isinstance(b, PythonBackend)


def test_get_backend_native() -> None:
    b = get_backend("native")
    assert isinstance(b, NativeBackend)


def test_get_backend_default() -> None:
    """Default backend is python."""
    b = get_backend()
    assert isinstance(b, PythonBackend)


def test_get_backend_unknown_raises() -> None:
    with pytest.raises(ValueError):
        get_backend("bogus")


def test_get_backend_case_insensitive() -> None:
    b = get_backend("Python")
    assert isinstance(b, PythonBackend)
    b = get_backend("NATIVE")
    assert isinstance(b, NativeBackend)


# ---------------------------------------------------------------------------
# BackendBase.dispatch
# ---------------------------------------------------------------------------


class _TrivialBackend(BackendBase):
    """Backend that just records which method was called."""

    @property
    def name(self) -> str:
        return "trivial"

    @property
    def available(self) -> bool:
        return True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def matmul_q4k(self, x, weight, n, k):
        self.calls.append("matmul_q4k")
        return np.zeros(n, dtype=np.float32)

    def matmul_q6k(self, x, weight, n, k):
        self.calls.append("matmul_q6k")
        return np.zeros(n, dtype=np.float32)

    def matmul_q8_0(self, x, weight, n, k):
        self.calls.append("matmul_q8_0")
        return np.zeros(n, dtype=np.float32)


def test_dispatch_routes_q4_k_to_matmul_q4k() -> None:
    b = _TrivialBackend()
    x = np.zeros(256, dtype=np.float32)
    w = np.zeros(256, dtype=np.float32)
    b.dispatch(x, w, 1, 256, "Q4_K")
    assert b.calls == ["matmul_q4k"]


def test_dispatch_routes_q6_k_to_matmul_q6k() -> None:
    b = _TrivialBackend()
    x = np.zeros(256, dtype=np.float32)
    w = np.zeros(256, dtype=np.float32)
    b.dispatch(x, w, 1, 256, "Q6_K")
    assert b.calls == ["matmul_q6k"]


def test_dispatch_routes_q8_0_to_matmul_q8_0() -> None:
    b = _TrivialBackend()
    x = np.zeros(32, dtype=np.float32)
    w = np.zeros(32, dtype=np.float32)
    b.dispatch(x, w, 1, 32, "Q8_0")
    assert b.calls == ["matmul_q8_0"]


def test_dispatch_falls_back_to_f32_matmul_for_unknown_quant() -> None:
    """Unknown quant (e.g. F32) falls back to weight @ x."""
    b = _TrivialBackend()
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    out = b.dispatch(x, w, 2, 3, "F32")
    # No method called
    assert b.calls == []
    # F32 path: w @ x
    np.testing.assert_array_equal(out, [1.0, 2.0])


def test_dispatch_falls_back_when_method_missing() -> None:
    """Backend without matmul_q6k falls back to F32 for Q6_K."""

    class _NoQ6K(BackendBase):
        @property
        def name(self) -> str:
            return "no_q6k"

        @property
        def available(self) -> bool:
            return True

        def matmul_q4k(self, x, weight, n, k):
            return np.zeros(n, dtype=np.float32)

        # Remove the inherited matmul_q6k so getattr returns None.
        # This mirrors a backend that genuinely doesn't support Q6_K.
        matmul_q6k = None  # type: ignore[assignment]

    b = _NoQ6K()
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    out = b.dispatch(x, w, 2, 3, "Q6_K")
    np.testing.assert_array_equal(out, [1.0, 2.0])


# ---------------------------------------------------------------------------
# PythonBackend raw dequant
# ---------------------------------------------------------------------------


def _make_q4_k_bytes(seed: int, n_blocks: int) -> bytes:
    import struct
    rng = np.random.default_rng(seed)
    out = bytearray()
    for _ in range(n_blocks):
        d = float(rng.uniform(0.01, 1.0))
        dmin = float(rng.uniform(0.0, 0.5))
        out += struct.pack("<ee", d, dmin)
        out += bytes(rng.integers(0, 64, size=12, dtype=np.uint8))
        out += bytes(rng.integers(0, 16, size=128, dtype=np.uint8))
    return bytes(out)


def _make_q6_k_bytes(seed: int, n_blocks: int) -> bytes:
    """Q6_K layout: ql[128] | qh[64] | scales[16] | d (FP16)."""
    import struct
    rng = np.random.default_rng(seed)
    out = bytearray()
    for _ in range(n_blocks):
        out += bytes(rng.integers(0, 16, size=128, dtype=np.uint8))
        out += bytes(rng.integers(0, 4, size=64, dtype=np.uint8))
        out += bytes(rng.integers(-127, 127, size=16, dtype=np.int8))
        d = float(rng.uniform(0.01, 1.0))
        out += struct.pack("<e", d)
    return bytes(out)


def _make_q8_0_bytes(seed: int, n_blocks: int) -> bytes:
    import struct
    rng = np.random.default_rng(seed)
    out = bytearray()
    for _ in range(n_blocks):
        d = float(rng.uniform(0.01, 1.0))
        out += struct.pack("<e", d)
        out += bytes(rng.integers(-127, 127, size=32, dtype=np.int8))
    return bytes(out)


def test_python_backend_dequant_q4_k() -> None:
    """PythonBackend.matmul_q4k produces a valid F32 vector."""
    from flatrun.dequant.gguf import dequant_q4_k
    n, k = 4, 256
    raw = np.frombuffer(_make_q4_k_bytes(0, n * k // 256), dtype=np.uint8)
    x = np.random.default_rng(0).standard_normal(k).astype(np.float32)
    w_f32 = dequant_q4_k(raw.tobytes(), (n, k), np.float32)
    out = PythonBackend().matmul_q4k(x, raw, n, k)
    expected = w_f32 @ x
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_python_backend_dequant_q6_k() -> None:
    """PythonBackend.matmul_q6k produces a valid F32 vector."""
    from flatrun.dequant.gguf import dequant_q6_k
    n, k = 4, 256
    raw = np.frombuffer(_make_q6_k_bytes(0, n * k // 256), dtype=np.uint8)
    x = np.random.default_rng(0).standard_normal(k).astype(np.float32)
    w_f32 = dequant_q6_k(raw.tobytes(), (n, k), np.float32)
    out = PythonBackend().matmul_q6k(x, raw, n, k)
    expected = w_f32 @ x
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_python_backend_dequant_q8_0() -> None:
    """PythonBackend.matmul_q8_0 produces a valid F32 vector."""
    from flatrun.dequant.gguf import dequant_q8_0
    n, k = 4, 32
    raw = np.frombuffer(_make_q8_0_bytes(0, n * k // 32), dtype=np.uint8)
    x = np.random.default_rng(0).standard_normal(k).astype(np.float32)
    w_f32 = dequant_q8_0(raw.tobytes(), (n, k), np.float32)
    out = PythonBackend().matmul_q8_0(x, raw, n, k)
    expected = w_f32 @ x
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_python_backend_unknown_quant_raises() -> None:
    """PythonBackend._dequant_raw rejects unknown quant tags."""
    py = PythonBackend()
    x = np.zeros(32, dtype=np.float32)
    w = np.zeros(32, dtype=np.uint8)
    with pytest.raises(ValueError):
        py._dequant_raw(w, 1, 32, "BOGUS_QUANT")


# ---------------------------------------------------------------------------
# NativeBackend fallback path
# ---------------------------------------------------------------------------


def test_native_backend_unavailable_falls_back_to_python() -> None:
    """When _C is unavailable, NativeBackend falls back to PythonBackend."""
    from flatrun.dequant.gguf import dequant_q4_k
    n = NativeBackend()
    real_available = n._native.available
    try:
        n._native.available = False
        # The supported_quants should now be empty
        assert n.supported_quants == set()
        # matmul_q4k should still work via the PythonBackend fallback
        n_rows, k = 4, 256
        raw = np.frombuffer(_make_q4_k_bytes(0, n_rows * k // 256), dtype=np.uint8)
        x = np.random.default_rng(0).standard_normal(k).astype(np.float32)
        w_f32 = dequant_q4_k(raw.tobytes(), (n_rows, k), np.float32)
        out = n.matmul_q4k(x, raw, n_rows, k)
        expected = w_f32 @ x
        np.testing.assert_allclose(out, expected, atol=1e-4)
    finally:
        n._native.available = real_available
