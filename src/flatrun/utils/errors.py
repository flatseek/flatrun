"""FlatRun exception hierarchy.

All errors raised by FlatRun derive from :class:`FlatRunError` so callers
can catch a single root type while still being able to distinguish the
specific failure mode if they care.
"""

from __future__ import annotations


class FlatRunError(Exception):
    """Root exception for every FlatRun error."""


class BackendError(FlatRunError):
    """A storage backend failed to open, parse, or read a file."""


class TensorNotFoundError(BackendError):
    """The requested tensor name is not present in the backend."""

    def __init__(self, name: str, backend: str) -> None:
        super().__init__(f"Tensor {name!r} not found in backend {backend!r}")
        self.name = name
        self.backend = backend


class MemoryError_(FlatRunError):
    """Raised when the memory manager cannot satisfy a request.

    Python's builtin ``MemoryError`` is reserved for the OS allocator, so
    the trailing underscore avoids the name clash.
    """


class LayerStreamingError(FlatRunError):
    """Raised when layer streaming hits an unrecoverable problem."""


class ConfigurationError(FlatRunError):
    """Invalid configuration was supplied to the runtime."""


__all__ = [
    "FlatRunError",
    "BackendError",
    "TensorNotFoundError",
    "MemoryError_",
    "LayerStreamingError",
    "ConfigurationError",
]