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


def test_scheduler_position_flags_match_layer_indices(synthetic_model) -> None:
    """``is_first`` / ``is_last`` / ``position`` must flip on the first
    and last *selected* layers, not on the original layer 0 / last.

    Drives a custom layer selection through the scheduler and
    verifies the flags are set on the first and last item in the
    selected list, regardless of which original model index they map
    to.
    """
    backend = open_safetensors(synthetic_model)
    try:
        manifest = build_manifest(backend)
        mgr = MemoryManager(backend, MemoryConfig(probe=None))
        # Pick a non-trivial subset: skip layer 0, pick layers 1, 2, 3.
        all_layers = list(manifest.layers)
        keep_indices = [1, 2, 3]
        selected = tuple(
            layer for layer in all_layers if layer.index in keep_indices
        )
        assert len(selected) == len(keep_indices)
        scheduler = LayerScheduler(mgr, selected)

        seen: list[tuple[int, bool, bool, int]] = []

        def cb(layer, handles):  # type: ignore[no-untyped-def]
            seen.append((layer.index, handles.is_first, handles.is_last, handles.position))

        scheduler.run(cb)

        # First selected layer is layer 1; it carries ``is_first``.
        assert seen[0] == (1, True, False, 0)
        # Last selected layer is layer 3; it carries ``is_last``.
        assert seen[-1] == (3, False, True, len(keep_indices) - 1)
        # Middle layers carry neither flag.
        for idx, is_first, is_last, position in seen[1:-1]:
            assert is_first is False
            assert is_last is False
        # Positions are dense (0..N-1).
        assert [s[3] for s in seen] == list(range(len(keep_indices)))
    finally:
        backend.close()


def test_scheduler_pre_post_bound_by_position(synthetic_model) -> None:
    """Pre/post-layer tensors follow the *selected* first/last layer,
    not the model's original layer 0 / last.

    With a custom selection that doesn't include the original
    layer 0, the embedding tensor must be loaded with the first
    selected layer, and ``model.norm.weight`` / ``lm_head.weight``
    with the last selected layer.
    """
    backend = open_safetensors(synthetic_model)
    try:
        manifest = build_manifest(backend)
        mgr = MemoryManager(backend, MemoryConfig(cache_entries=None, probe=None))
        # Pick a subset that doesn't include layer 0.
        all_layers = list(manifest.layers)
        keep_indices = [1, 2, 3]
        selected = tuple(
            layer for layer in all_layers if layer.index in keep_indices
        )
        # Use the real pre/post tensors from the model so the
        # scheduler can actually load them.
        pre_names = [n for n in manifest.pre_layer if n]
        post_names = [n for n in manifest.post_layer if n]
        scheduler = LayerScheduler(
            mgr, selected,
            pre_layer_names=pre_names,
            post_layer_names=post_names,
        )
        seen: list[tuple[int, bool, bool, frozenset[str]]] = []

        def cb(layer, handles):  # type: ignore[no-untyped-def]
            # ``handles.keys()`` is the actual set of loaded tensors;
            # ``names_in_order()`` also includes pre/post names for
            # bookkeeping even when they aren't loaded on this layer.
            seen.append(
                (
                    layer.index,
                    handles.is_first,
                    handles.is_last,
                    frozenset(handles.keys()),
                )
            )

        scheduler.run(cb)

        # First visited layer has is_first=True, the real "pre-layer"
        # tensors are loaded with it.
        first_idx, first_is_first, _, first_names = seen[0]
        assert first_is_first is True
        assert first_idx == 1
        for n in pre_names:
            assert n in first_names, f"pre tensor {n} missing on first selected layer"

        # Last visited layer has is_last=True, the real post-layer
        # tensors are loaded with it.
        last_idx, _, last_is_last, last_names = seen[-1]
        assert last_is_last is True
        assert last_idx == 3
        for n in post_names:
            assert n in last_names, f"post tensor {n} missing on last selected layer"

        # Middle layers carry neither flag and don't carry pre/post.
        for idx, is_first, is_last, names in seen[1:-1]:
            assert is_first is False
            assert is_last is False
            for n in pre_names:
                assert n not in names, f"pre tensor {n} leaked onto middle layer"
            for n in post_names:
                assert n not in names, f"post tensor {n} leaked onto middle layer"
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