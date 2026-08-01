"""Public Python API for flatrun_native.

This package is an optional accelerator. The core Flatrun runtime
continues to work without it; the import is wrapped in a try/except
so users without the C++ toolchain can still install the parent
package.

Typical use:

    from flatrun_native import NativeBackend

    backend = NativeBackend()
    if backend.available:
        out = backend.matmul_q4k(x, weight)
    else:
        # fall back to the numpy path
        ...

The backend is intentionally a thin wrapper around the C++ extension
- all kernel implementations live in the ``_C`` module and the
Python layer only adds shape validation, dtype alignment, and the
Python↔numpy ↔C++ data marshalling.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "NativeBackend",
    "is_available",
    "version",
    "Q4K_GEMM_BLOCK",
]

# Block size of the Q4_K format — kept in sync with the C++ header.
Q4K_GEMM_BLOCK = 256

# ---------------------------------------------------------------------------
# Extension loader
# ---------------------------------------------------------------------------
#
# We try ``flatrun_native._C`` first (the in-tree build). If the
# wrapper package was installed without the C++ extension, the import
# raises ``ModuleNotFoundError`` and we fall back to a stub that fails
# loudly on every call. The Python code in the runtime uses
# :func:`is_available` to decide whether to dispatch to the native
# backend or the numpy fallback.

_C: Any = None
_LOAD_ERROR: BaseException | None = None

try:
    from flatrun_native import _C  # noqa: F401, E402
except ImportError as exc:
    _LOAD_ERROR = exc
    _C = None


def is_available() -> bool:
    """Return ``True`` when the C++ extension imported successfully."""
    return _C is not None


def version() -> str:
    """Return the build identifier, or a fallback string."""
    if _C is None:
        return f"flatrun_native (unavailable: {_LOAD_ERROR})"
    return _C.version()


def is_neon() -> bool:
    """Return ``True`` when the C++ extension was built with NEON intrinsics."""
    if _C is None:
        return False
    return bool(_C.is_neon())


# ---------------------------------------------------------------------------
# NativeBackend
# ---------------------------------------------------------------------------


class NativeBackend:
    """High-level facade over the C++ kernels.

    The backend is constructed cheaply and is safe to share across
    threads. Each call hands the caller's input arrays to the C++
    extension; no state is retained between calls.

    Attributes
    ----------
    available : bool
        True if the C++ extension loaded. False otherwise. Methods
        on a non-available backend raise :class:`RuntimeError`.
    """

    def __init__(self) -> None:
        self.available = is_available()
        self._neon = is_neon() if self.available else False

    def matmul_q4k(
        self,
        x: "numpy.ndarray",
        weight: "numpy.ndarray",
        n: int,
        k: int,
    ) -> "numpy.ndarray":
        """Fused Q4_K dequant + matrix-vector / matrix-matrix multiply.

        Accepts both 1-D ``x`` of shape ``(k,)`` (single-token decode)
        and 2-D ``x`` of shape ``(seq, k)`` (prefill / batched decode).
        The single-token call delegates to the C++ ``matmul_q4_k``;
        the batched call uses ``matmul_q4_k_batched`` which dispatches
        the kernel ``seq`` times inside C++ - eliminating the
        pybind11 round-trip per token that dominated the profile when
        the forwarder called the per-token API in a Python loop.

        Parameters
        ----------
        x : np.ndarray
            Float32 input activations of shape ``(k,)`` or ``(seq, k)``.
        weight : np.ndarray
            Raw Q4_K block bytes (uint8, length ``n * k/256 * 144``).
        n : int
            Output feature count (rows of the weight matrix).
        k : int
            Input feature count (cols of the weight matrix); must be a
            multiple of 256.

        Returns
        -------
        np.ndarray
            Float32 output of shape ``(n,)`` (1-D input) or ``(seq, n)``
            (2-D input). The caller is responsible for any
            post-processing (bias add, residual, ...).
        """
        if not self.available:
            raise RuntimeError(
                "flatrun_native._C is not available - rebuild with "
                "'pip install -e .' on a host with pybind11 installed."
            )
        import numpy as np
        x_c = np.ascontiguousarray(x, dtype=np.float32)
        w_c = np.ascontiguousarray(weight, dtype=np.uint8)
        if x_c.ndim == 1:
            if x_c.shape[0] != k:
                raise ValueError(f"x.shape[0]={x_c.shape[0]} != k={k}")
            return _C.matmul_q4_k(x_c, w_c, int(n), int(k))
        if x_c.ndim == 2:
            return _C.matmul_q4_k_batched(x_c, w_c, int(n), int(k))
        raise ValueError(f"x must be 1-D or 2-D, got shape {x_c.shape}")

    def matmul_q8_0(
        self,
        x: "numpy.ndarray",
        weight: "numpy.ndarray",
        n: int,
        k: int,
    ) -> "numpy.ndarray":
        """Fused Q8_0 dequant + matrix-vector / matrix-matrix multiply.

        Q8_0 is the format SmolLM2-360M and many LM Studio defaults ship
        in (e.g. ``*-Q8_0.gguf``). Without this kernel the Q4_K backend
        fell back to the numpy path for every SmolLM2 tensor.

        Accepts 1-D ``(k,)`` or 2-D ``(seq, k)`` ``x``; the 2-D form
        uses the batched C++ path that eliminates per-token overhead.
        """
        if not self.available:
            raise RuntimeError("flatrun_native._C is not available")
        import numpy as np
        x_c = np.ascontiguousarray(x, dtype=np.float32)
        w_c = np.ascontiguousarray(weight, dtype=np.uint8)
        if x_c.ndim == 1:
            if x_c.shape[0] != k:
                raise ValueError(f"x.shape[0]={x_c.shape[0]} != k={k}")
            return _C.matmul_q8_0(x_c, w_c, int(n), int(k))
        if x_c.ndim == 2:
            return _C.matmul_q8_0_batched(x_c, w_c, int(n), int(k))
        raise ValueError(f"x must be 1-D or 2-D, got shape {x_c.shape}")

    def matmul_q6_k(
        self,
        x: "numpy.ndarray",
        weight: "numpy.ndarray",
        n: int,
        k: int,
    ) -> "numpy.ndarray":
        """Fused Q6_K dequant + matrix-vector / matrix-matrix multiply.

        Q6_K appears in mixed-precision Q4_K_M checkpoints (Qwen3-0.6B
        uses Q6_K for v_proj / down_proj on alternating layers).
        Without this kernel a single non-Q4_K tensor would force the
        whole layer back to the numpy path.
        """
        if not self.available:
            raise RuntimeError("flatrun_native._C is not available")
        import numpy as np
        x_c = np.ascontiguousarray(x, dtype=np.float32)
        w_c = np.ascontiguousarray(weight, dtype=np.uint8)
        if x_c.ndim == 1:
            if x_c.shape[0] != k:
                raise ValueError(f"x.shape[0]={x_c.shape[0]} != k={k}")
            return _C.matmul_q6_k(x_c, w_c, int(n), int(k))
        if x_c.ndim == 2:
            return _C.matmul_q6_k_batched(x_c, w_c, int(n), int(k))
        raise ValueError(f"x must be 1-D or 2-D, got shape {x_c.shape}")

    def dequant_q4k(
        self,
        weight: "numpy.ndarray",
        shape: tuple[int, int],
    ) -> "numpy.ndarray":
        """Pure Q4_K dequant wrapper (test / validation path).

        Returns an F32 ndarray of the given shape. The forwarder never
        uses this directly — :meth:`matmul_q4k` is the hot path.
        """
        if not self.available:
            raise RuntimeError(
                "flatrun_native._C is not available"
            )
        import numpy as np
        w_c = np.ascontiguousarray(weight, dtype=np.uint8)
        return _C.dequant_q4_k(w_c, shape)
