"""Tests for the LayerScheduler."""

from __future__ import annotations

import pytest

from flatrun.backend.safetensor import open_safetensors
from flatrun.model.manifest import build_manifest
from flatrun.runtime.memory import MemoryConfig, MemoryManager
from flatrun.runtime.scheduler import LayerScheduler
from flatrun.utils.errors import LayerStreamingError


def test_run_visits_every_layer(synthetic_model) -> None:
    backend = open_safetensors(synthetic_model)
    try:
        manifest = build_manifest(backend)
        mgr = MemoryManager(backend, MemoryConfig(probe=None))
        scheduler = LayerScheduler(mgr, manifest.layers)

        seen: list[int] = []

        def cb(layer, handles):  # type: ignore[no-untyped-def]
            seen.append(layer.index)
            return f"layer-{layer.index}"

        results = scheduler.run(cb)
        assert seen == sorted(seen)
        assert results == [f"layer-{i}" for i in seen]
        assert scheduler.stats.layers_executed == len(manifest.layers)
    finally:
        backend.close()


def test_step_single_layer(synthetic_model) -> None:
    backend = open_safetensors(synthetic_model)
    try:
        manifest = build_manifest(backend)
        mgr = MemoryManager(backend, MemoryConfig(probe=None))
        scheduler = LayerScheduler(mgr, manifest.layers)
        result = scheduler.step(0, lambda layer, handles: "ok")
        assert result == "ok"
    finally:
        backend.close()


def test_step_out_of_range(synthetic_model) -> None:
    backend = open_safetensors(synthetic_model)
    try:
        manifest = build_manifest(backend)
        mgr = MemoryManager(backend, MemoryConfig(probe=None))
        scheduler = LayerScheduler(mgr, manifest.layers)
        with pytest.raises(LayerStreamingError):
            scheduler.step(999, lambda l, h: None)
    finally:
        backend.close()


def test_prefetch_hook_invoked(synthetic_model) -> None:
    backend = open_safetensors(synthetic_model)
    try:
        manifest = build_manifest(backend)
        mgr = MemoryManager(backend, MemoryConfig(probe=None))
        scheduler = LayerScheduler(mgr, manifest.layers)
        calls: list[int] = []
        scheduler.set_prefetch_hook(lambda layer: calls.append(layer.index))
        scheduler.run(lambda l, h: None)
        # Hook is called once per layer (before acquire).
        assert len(calls) == len(manifest.layers)
    finally:
        backend.close()


def test_scheduler_releases_after_compute(synthetic_model) -> None:
    backend = open_safetensors(synthetic_model)
    try:
        manifest = build_manifest(backend)
        mgr = MemoryManager(backend, MemoryConfig(cache_entries=None, probe=None))
        scheduler = LayerScheduler(mgr, manifest.layers)

        def cb(layer, handles):  # type: ignore[no-untyped-def]
            # Cache should hold exactly this layer's tensors during compute.
            current = set(handles.names_in_order())
            assert current.issubset(set(mgr.cached_names()))
            return current

        results = scheduler.run(cb)
        assert len(results) == len(manifest.layers)
        # After the run the manager has released every layer.
        assert mgr.stats().cache_entries == 0
    finally:
        backend.close()


def test_empty_layers_raises() -> None:
    from flatrun.backend.base import StorageBackend
    from flatrun.utils.types import TensorKey

    class _Empty(StorageBackend):
        @property
        def name(self):  # type: ignore[no-untyped-def]
            return "empty"

        @property
        def root(self):  # type: ignore[no-untyped-def]
            from pathlib import Path
            return Path("/dev/null")

        def open(self) -> None: ...
        def close(self) -> None: ...
        def list_tensors(self):  # type: ignore[no-untyped-def]
            return iter([])
        def get_metadata(self, name):  # type: ignore[no-untyped-def]
            raise TensorNotFoundError(name, "empty")
        @property
        def supports_mmap(self) -> bool:
            return True
        def open_handle(self, name):  # type: ignore[no-untyped-def]
            raise TensorNotFoundError(name, "empty")

    mgr = MemoryManager(_Empty(), MemoryConfig(probe=None))
    with pytest.raises(LayerStreamingError):
        LayerScheduler(mgr, [])

    _ = TensorKey  # silence unused import warning in some tools