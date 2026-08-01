"""Public API for flatrun_native.

Re-export from the python subpackage so ``from flatrun_native import
NativeBackend`` works without an extra import path.
"""

from __future__ import annotations

from .python import (
    NativeBackend,
    Q4K_GEMM_BLOCK,
    is_available,
    is_neon,
    version,
)

__all__ = [
    "NativeBackend",
    "Q4K_GEMM_BLOCK",
    "is_available",
    "is_neon",
    "version",
]
