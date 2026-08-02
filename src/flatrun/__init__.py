"""FlatRun - a streaming inference runtime for LLMs that don't fit in RAM.

FlatRun is the equivalent of a virtual memory subsystem for neural
network weights: model files live on disk, only the current layer is
resident in RAM at any time, and existing formats (SafeTensors, GGUF)
act as storage backends rather than execution engines.

The top-level package re-exports the public surface so typical imports
look like::

    from flatrun import InferenceRuntime, load_huggingface

See :mod:`flatrun.backend`, :mod:`flatrun.runtime`, and
:mod:`flatrun.model` for the implementation details.
"""

from __future__ import annotations

from .backend import (
    MultiBackend,
    SafeTensorBackend,
    StorageBackend,
    default_registry,
    open_safetensors,
)
from .model import (
    HFConfig,
    LoadedModel,
    ManifestBuilder,
    ModelManifest,
    Qwen2Config,
    build_manifest,
    load_huggingface,
    make_qwen2_forwarder,
)
from .runtime import (
    ForwardFn,
    InferenceRuntime,
    KVCache,
    LayerScheduler,
    MemoryConfig,
    MemoryManager,
    ModelExecutor,
    RuntimeConfig,
    RuntimeStats,
    StreamingExecutor,
    TensorHandle,
    TensorView,
    TokenStep,
)
from .utils import (
    BackendError,
    ConfigurationError,
    EvictionPolicy,
    FlatRunError,
    LayerDescriptor,
    LayerStreamingError,
    MemoryError_,
    MemoryProbe,
    MemoryStats,
    TensorKey,
    TensorMetadata,
    TensorNotFoundError,
    default_probe,
)

__version__ = "0.1.1"

__all__ = [
    "__version__",
    "BackendError",
    "ConfigurationError",
    "EvictionPolicy",
    "FlatRunError",
    "ForwardFn",
    "HFConfig",
    "InferenceRuntime",
    "KVCache",
    "LayerDescriptor",
    "LayerScheduler",
    "LayerStreamingError",
    "LoadedModel",
    "ManifestBuilder",
    "MemoryConfig",
    "MemoryError_",
    "MemoryManager",
    "MemoryProbe",
    "MemoryStats",
    "ModelExecutor",
    "ModelManifest",
    "MultiBackend",
    "Qwen2Config",
    "RuntimeConfig",
    "RuntimeStats",
    "SafeTensorBackend",
    "StorageBackend",
    "StreamingExecutor",
    "TensorHandle",
    "TensorKey",
    "TensorMetadata",
    "TensorNotFoundError",
    "TensorView",
    "TokenStep",
    "build_manifest",
    "default_probe",
    "default_registry",
    "load_huggingface",
    "make_qwen2_forwarder",
    "open_safetensors",
]