"""Populate the default backend registry.

Called from :mod:`flatrun.backend.registry` on first use. Importing the
concrete backend modules at this point keeps the registry table
authoritative without forcing every test to import every backend.
"""

from __future__ import annotations

from .registry import BackendRegistry


def populate(reg: BackendRegistry) -> None:
    # Importing here pulls in numpy / safetensors lazily.
    from .gguf import GGUFBackend
    from .safetensor import SafeTensorBackend

    reg.register_suffix("safetensors", SafeTensorBackend)
    reg.register_name("safetensors", SafeTensorBackend)
    reg.register_name("st", SafeTensorBackend)

    reg.register_suffix("gguf", GGUFBackend)
    reg.register_name("gguf", GGUFBackend)


__all__ = ["populate"]