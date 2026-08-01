"""Execution backend facade.

The runtime's forwarder uses a single object — the "backend" — to
perform the hot operations (today, the projection matmul). The class
exposes a small, stable interface; the implementation is either the
pure-Python numpy path (the default) or, when the optional
``flatrun_native`` C++ extension is available, the SIMD-accelerated
fused dequant + matmul path.

The user's choice is encoded in the ``--backend`` CLI flag:

* ``--backend python`` — always uses the numpy path. This is the
  default if the native backend is unavailable.
* ``--backend native`` — uses the C++ kernels when available. Falls
  back to the python path with a warning if the extension can't be
  loaded.

The backend is intentionally narrow: it knows how to multiply an
FP32 activation vector x by a quantised weight matrix and return the
FP32 output. Everything else (tokenization, sampling, KV cache,
residual, embedding) stays in the forwarder.

Supported quantisation formats:

* ``Q4_K`` — 4-bit K-quant, 256 elements / 144-byte block.
* ``Q6_K`` — 6-bit K-quant, 256 elements / 210-byte block.
* ``Q8_0`` — 8-bit symmetric, 32 elements / 34-byte block.

The native backend implements all three in C++/NEON; the python
backend implements the equivalent dequant + matmul via numpy for
each.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


# Map every quantisation the backend knows about to the corresponding
# method name. The dispatcher in the forwarder consults this table
# (via :meth:`BackendBase.dispatch`) instead of a long if/elif chain.
_QUANT_DISPATCH: dict[str, str] = {
    "Q4_K": "matmul_q4k",
    "Q6_K": "matmul_q6k",
    "Q8_0": "matmul_q8_0",
}


def _quant_method(quant: str | None) -> str | None:
    """Return the backend method name that handles ``quant`` or ``None``."""
    if quant is None:
        return None
    return _QUANT_DISPATCH.get(quant)


class BackendBase(ABC):
    """Abstract base class for projection backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier (``"python"`` or ``"native"``)."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """True when the backend is ready to dispatch calls."""

    @property
    def supported_quants(self) -> set[str]:
        """Quant types the backend can handle natively.

        The python backend supports every format via its
        ``dequant + matmul`` path; the native backend reports the
        subset its compiled kernels cover. The forwarder uses this to
        decide whether to dispatch or fall back.
        """
        return set(_QUANT_DISPATCH.keys())

    def matmul_q4k(
        self,
        x: np.ndarray,
        weight_handle: Any,
        n: int,
        k: int,
    ) -> np.ndarray:
        """Compute ``out = w @ x`` for a Q4_K weight.

        Parameters
        ----------
        x : np.ndarray
            Shape (k,) FP32 input activations.
        weight_handle : any
            A tensor handle whose underlying bytes are the Q4_K
            packed payload of a (n, k) row-major matrix.
        n : int
            Output feature count (rows of the weight matrix).
        k : int
            Input feature count (cols of the weight matrix).

        Returns
        -------
        np.ndarray
            Shape (n,) FP32 output.
        """
        raise NotImplementedError

    def matmul_q6k(
        self,
        x: np.ndarray,
        weight: Any,
        n: int,
        k: int,
    ) -> np.ndarray:
        """Compute ``out = w @ x`` for a Q6_K weight (256-element blocks)."""
        raise NotImplementedError

    def matmul_q8_0(
        self,
        x: np.ndarray,
        weight: Any,
        n: int,
        k: int,
    ) -> np.ndarray:
        """Compute ``out = w @ x`` for a Q8_0 weight (32-element blocks)."""
        raise NotImplementedError

    def dispatch(
        self,
        x: np.ndarray,
        weight: Any,
        n: int,
        k: int,
        quant: str | None,
    ) -> np.ndarray:
        """Dispatch a matmul to the right per-quant method.

        ``quant`` is the on-disk quantisation tag (``"Q4_K"``,
        ``"Q6_K"``, ``"Q8_0"``, ``None`` for F32). When the backend
        supports ``quant`` the corresponding method is called;
        otherwise the call falls back to F32 ``weight @ x`` (the
        dequant-cache path produces an F32 buffer, so any unknown
        quant lands here).
        """
        method_name = _quant_method(quant)
        if method_name is None:
            # Unknown / non-quantised weight: assume F32.
            return weight @ x
        method = getattr(self, method_name, None)
        if method is None:
            # Backend doesn't implement this quant. Treat as F32.
            return weight @ x
        return method(x, weight, n, k)


# ---------------------------------------------------------------------------
# Python (numpy) backend
# ---------------------------------------------------------------------------


class PythonBackend(BackendBase):
    """Pure-numpy / pure-Python backend. Always available."""

    @property
    def name(self) -> str:
        return "python"

    @property
    def available(self) -> bool:
        return True

    @property
    def supported_quants(self) -> set[str]:
        # numpy dequant path handles every GGUF type we ship.
        return set(_QUANT_DISPATCH.keys()) | {"F32", "F16", "MLX_4BIT"}

    def _dequant_raw(self, weight: Any, n: int, k: int, quant: str) -> np.ndarray:
        """Decode ``weight`` (raw bytes or a handle) to F32 ``(n, k)``.

        Single dispatch site for the python path so adding a new
        quant is a one-line change.
        """
        from flatrun.dequant.gguf import (
            dequant_q4_k,
            dequant_q6_k,
            dequant_q8_0,
        )
        if hasattr(weight, "as_numpy"):
            raw = weight.as_numpy()
        else:
            raw = weight
        if quant == "Q4_K":
            return dequant_q4_k(raw, (n, k), np.float32)
        if quant == "Q6_K":
            return dequant_q6_k(raw, (n, k), np.float32)
        if quant == "Q8_0":
            return dequant_q8_0(raw, (n, k), np.float32)
        raise ValueError(f"python backend: unknown quant {quant!r}")

    def matmul_q4k(self, x, weight, n, k):
        w = self._dequant_raw(weight, n, k, "Q4_K")
        return w @ x

    def matmul_q6k(self, x, weight, n, k):
        w = self._dequant_raw(weight, n, k, "Q6_K")
        return w @ x

    def matmul_q8_0(self, x, weight, n, k):
        w = self._dequant_raw(weight, n, k, "Q8_0")
        return w @ x


# ---------------------------------------------------------------------------
# Native (C++/NEON) backend
# ---------------------------------------------------------------------------


class NativeBackend(BackendBase):
    """Wrapper around the optional ``flatrun_native`` C++ extension.

    Implements per-quant fused dequant + matmul kernels for Q4_K,
    Q6_K, and Q8_0. When the extension is not built, or when the
    kernel rejects an unusual layout (raises RuntimeError), the call
    falls back to the pure-python path so ``--backend native`` is
    always safe.
    """

    def __init__(self) -> None:
        from flatrun_native import NativeBackend as _NativeBackend
        self._native = _NativeBackend()
        self._fallback = PythonBackend()

    @property
    def name(self) -> str:
        return "native"

    @property
    def available(self) -> bool:
        return bool(self._native.available)

    @property
    def supported_quants(self) -> set[str]:
        if not self._native.available:
            return set()
        return {"Q4_K", "Q6_K", "Q8_0"}

    @staticmethod
    def _raw(weight: Any) -> np.ndarray:
        """Return a contiguous uint8 ndarray view of the weight bytes."""
        if hasattr(weight, "as_numpy"):
            raw = weight.as_numpy()
        else:
            raw = np.ascontiguousarray(weight, dtype=np.uint8)
        return raw

    def _try_kernel(self, kernel, fallback_name: str, x, weight, n, k):
        """Call ``kernel(x, raw, n, k)`` or fall back on RuntimeError."""
        if not self._native.available:
            return getattr(self._fallback, fallback_name)(x, weight, n, k)
        try:
            raw = self._raw(weight)
            return kernel(x, raw, n, k)
        except RuntimeError:
            # Kernel rejected this layout. Fall back to numpy.
            return getattr(self._fallback, fallback_name)(x, weight, n, k)

    def matmul_q4k(self, x, weight, n, k):
        return self._try_kernel(self._native.matmul_q4k, "matmul_q4k", x, weight, n, k)

    def matmul_q6k(self, x, weight, n, k):
        return self._try_kernel(self._native.matmul_q6_k, "matmul_q6k", x, weight, n, k)

    def matmul_q8_0(self, x, weight, n, k):
        return self._try_kernel(self._native.matmul_q8_0, "matmul_q8_0", x, weight, n, k)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[BackendBase]] = {
    "python": PythonBackend,
    "native": NativeBackend,
}


def get_backend(name: str = "python") -> BackendBase:
    """Return a backend instance for the requested name.

    Parameters
    ----------
    name : str
        ``"python"`` or ``"native"``. ``"native"`` falls back to
        ``"python"`` with a warning if the C++ extension is not
        available.

    Returns
    -------
    BackendBase
        A ready-to-use backend instance.

    Raises
    ------
    ValueError
        If ``name`` is not a recognised backend identifier.
    """
    name = (name or "python").lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown backend {name!r}; expected one of {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name]()


__all__ = [
    "BackendBase",
    "PythonBackend",
    "NativeBackend",
    "get_backend",
]
