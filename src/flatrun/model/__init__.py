"""Model-aware glue: manifest, HuggingFace loader, and the Qwen2 / Llama
reference forwarder."""

from .huggingface import HFConfig, LoadedModel, load_huggingface
from .manifest import (
    DEFAULT_LAYER_PATTERN,
    DEFAULT_NON_LAYER_TENSORS,
    ManifestBuilder,
    ModelManifest,
    build_manifest,
)
from .qwen2 import Qwen2Config, make_qwen2_forwarder

__all__ = [
    "DEFAULT_LAYER_PATTERN",
    "DEFAULT_NON_LAYER_TENSORS",
    "HFConfig",
    "LoadedModel",
    "ManifestBuilder",
    "ModelManifest",
    "Qwen2Config",
    "build_manifest",
    "load_huggingface",
    "make_qwen2_forwarder",
]
