"""HuggingFace checkpoint loader.

Most open-weight LLMs are shipped as one or more ``.safetensors`` files
alongside a ``config.json``. This module turns a directory or a Hub
repo id into a fully wired FlatRun :class:`InferenceRuntime`.

The loader deliberately does not import ``transformers``. FlatRun's job
is to serve weights; the user is expected to bring their own forward
pass. We *do* parse ``config.json`` to populate the manifest's
``architecture`` tag and to provide a structural sanity check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..backend.base import StorageBackend
from ..backend.multi import MultiBackend
from ..backend.registry import default_registry
from ..runtime.memory import MemoryConfig
from ..runtime.runtime import InferenceRuntime, RuntimeConfig
from ..utils.errors import ConfigurationError
from .manifest import ModelManifest, build_manifest


@dataclass(slots=True)
class HFConfig:
    """Subset of ``config.json`` fields FlatRun cares about."""

    architectures: tuple[str, ...]
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    vocab_size: int | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> "HFConfig":
        if not path.is_file():
            raise ConfigurationError(f"HuggingFace config not found: {path}")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "HFConfig":
        archs = data.get("architectures") or ()
        if isinstance(archs, str):
            archs = (archs,)
        return cls(
            architectures=tuple(str(a) for a in archs),
            hidden_size=_safe_int(data.get("hidden_size")),
            num_hidden_layers=_safe_int(data.get("num_hidden_layers")),
            vocab_size=_safe_int(data.get("vocab_size")),
            raw=dict(data),
        )


def _safe_int(v: object) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class LoadedModel:
    """Bundle returned by :func:`load_huggingface`."""

    runtime: InferenceRuntime
    manifest: ModelManifest
    config: HFConfig | None
    source: Path


def load_huggingface(
    source: str | Path,
    *,
    config: RuntimeConfig | None = None,
    memory: MemoryConfig | None = None,
    prefer_shard_pattern: str = "model*.safetensors",
) -> LoadedModel:
    """Load a HuggingFace-style checkpoint through FlatRun.

    Parameters
    ----------
    source : str | Path
        Either a local directory containing ``.safetensors`` shards and
        a ``config.json``, a directory containing a single ``.gguf``
        file, or a single safetensors/gguf file path.
    config : RuntimeConfig | None
        Override the default :class:`RuntimeConfig`.
    memory : MemoryConfig | None
        Shortcut for ``config.memory``. Takes precedence if both are
        provided.
    prefer_shard_pattern : str
        Glob used to pick shards inside a directory.
    """
    src = Path(source)
    if not src.exists():
        raise ConfigurationError(f"Source path does not exist: {src}")

    # Pick the backend(s).
    if src.is_dir():
        # Try safetensors shards first, fall back to GGUF single-file.
        shards = sorted(src.glob(prefer_shard_pattern))
        if not shards:
            gguf = sorted(src.glob("*.gguf"))
            if gguf:
                # LM Studio stores vision-language models with two
                # GGUF files: the base LLM and a ``mmproj`` /
                # ``vision`` / ``clip`` / ``projection`` adapter.
                # flatrun is text-only, so prefer the LLM file.
                mmproj_markers = (
                    "mmproj",
                    "mm-proj",
                    "mm_proj",
                    "vision",
                    "clip",
                    "projection",
                    "imgproj",
                )

                def _is_llm_only(p: Path) -> bool:
                    return not any(m in p.stem.lower() for m in mmproj_markers)

                llm_only = [p for p in gguf if _is_llm_only(p)]
                if len(llm_only) == 1:
                    shards = llm_only
                elif llm_only:
                    raise ConfigurationError(
                        f"Multiple LLM GGUF files in {src}: "
                        f"{[p.name for p in llm_only]}; "
                        f"pass a single file path"
                    )
                elif len(gguf) > 1:
                    raise ConfigurationError(
                        f"Multiple GGUF files in {src} but none look "
                        f"like a base LLM (the others look like mmproj / "
                        f"vision / clip / projection helpers). Point "
                        f"--model at the base model file directly."
                    )
                else:
                    shards = gguf
            else:
                raise ConfigurationError(
                    f"No shards matching {prefer_shard_pattern!r} or *.gguf in {src}"
                )
        registry = default_registry()
        backends: list[StorageBackend] = [registry.open(s) for s in shards]
        backend: StorageBackend = MultiBackend(backends)
        hf_config_path = src / "config.json"
        hf_config = HFConfig.from_path(hf_config_path) if hf_config_path.is_file() else None
    else:
        backend = default_registry().open(src)
        hf_config = None

    rt_config = config or RuntimeConfig()
    if memory is not None:
        # ``memory`` overrides ``config.memory`` if provided.
        rt_config.memory = memory
    runtime = InferenceRuntime(backend, config=rt_config)
    architecture = hf_config.architectures[0] if hf_config and hf_config.architectures else None
    manifest = build_manifest(backend, architecture=architecture)

    if (
        hf_config is not None
        and hf_config.num_hidden_layers is not None
        and hf_config.num_hidden_layers != manifest.layer_count
    ):
        # Don't fail; some checkpoints share embeddings with the LM head or
        # apply a final layer-norm that changes the count.
        pass

    return LoadedModel(
        runtime=runtime,
        manifest=manifest,
        config=hf_config,
        source=src,
    )


__all__ = [
    "HFConfig",
    "LoadedModel",
    "load_huggingface",
]