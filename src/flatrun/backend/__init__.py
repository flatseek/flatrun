"""Storage backend abstraction and shipped implementations.

Concrete backends live in submodules. They are wired into the
:func:`default_registry` on first import via :mod:`._populate`, so a
typical user never imports a backend class directly::

    from flatrun.backend import default_registry

    backend = default_registry().open("model.safetensors")
    handle = backend.open_handle("model.embed_tokens.weight")
"""

from .base import StorageBackend
from .multi import MultiBackend
from .registry import (
    BackendFactory,
    BackendRegistry,
    default_registry,
    register,
)
from .safetensor import SafeTensorBackend, open_safetensors

__all__ = [
    "BackendFactory",
    "BackendRegistry",
    "MultiBackend",
    "SafeTensorBackend",
    "StorageBackend",
    "default_registry",
    "open_safetensors",
    "register",
]
