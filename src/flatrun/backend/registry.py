"""Backend registry - file-extension and metadata driven dispatch.

The registry decouples the runtime from concrete backend implementations
so additional storage formats can be plugged in at runtime without
modifying the core. The default registry includes SafeTensors; GGUF and
remote backends register themselves when their modules are imported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .base import StorageBackend


# Type alias for backend factories.
BackendFactory = Callable[[Path], StorageBackend]


class BackendRegistry:
    """Registry mapping file suffixes and format hints to backend classes."""

    def __init__(self) -> None:
        self._suffixes: dict[str, BackendFactory] = {}
        self._names: dict[str, BackendFactory] = {}

    def register_suffix(self, suffix: str, factory: BackendFactory) -> None:
        self._suffixes[suffix.lower().lstrip(".")] = factory

    def register_name(self, name: str, factory: BackendFactory) -> None:
        self._names[name.lower()] = factory

    def resolve(self, path: Path, *, hint: str | None = None) -> BackendFactory:
        """Pick a factory for ``path``.

        The lookup order is:

        1. ``hint`` - explicit format identifier from the caller.
        2. Suffix match on the file name (``".safetensors"``, ``".gguf"``).
        3. Suffix match on ``".bin"`` (legacy torch checkpoint).
        """
        if hint is not None:
            key = hint.lower()
            if key in self._names:
                return self._names[key]
        suffix = path.suffix.lower().lstrip(".")
        if suffix in self._suffixes:
            return self._suffixes[suffix]
        if path.is_dir():
            # Look inside the directory for a single shard.
            for child in sorted(path.iterdir()):
                if child.is_file():
                    return self.resolve(child)
        raise LookupError(f"No backend registered for path {path!r} (hint={hint!r})")

    def open(self, path: Path, *, hint: str | None = None) -> StorageBackend:
        factory = self.resolve(path, hint=hint)
        return factory(path)


# Global default registry. Users are free to build their own registry if
# they want a sandboxed environment.
_default_registry = BackendRegistry()


def default_registry() -> BackendRegistry:
    """Return the process-wide default :class:`BackendRegistry`.

    The default registry is populated lazily on first access to avoid
    forcing every backend module to be imported eagerly.
    """
    from . import _populate  # local import to dodge cycles

    _populate.populate(_default_registry)
    return _default_registry


def register(suffix: str | None = None, *, name: str | None = None) -> Callable[[BackendFactory], BackendFactory]:
    """Decorator that registers a backend factory under ``suffix`` and/or ``name``."""

    def deco(factory: BackendFactory) -> BackendFactory:
        reg = default_registry()
        if suffix is not None:
            reg.register_suffix(suffix, factory)
        if name is not None:
            reg.register_name(name, factory)
        return factory

    return deco


__all__ = ["BackendFactory", "BackendRegistry", "default_registry", "register"]