"""Layer manifest - groups tensors into layers.

A :class:`ModelManifest` translates the flat list of tensors that a
storage backend exposes into a sequence of :class:`LayerDescriptor`
objects that the scheduler can stream. FlatRun ships a generic
manifest builder that supports the conventional HuggingFace naming
pattern (``model.layers.{i}.self_attn.q_proj.weight``).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from ..backend.base import StorageBackend
from ..utils.errors import ConfigurationError
from ..utils.types import LayerDescriptor, TensorKey


# Default regex for decoder layers in HuggingFace checkpoints.
# Qwen3.5 multimodal checkpoints nest the text decoder under a
# leading ``language_model.`` segment. The optional ``<segment>.`` prefix
# accommodates that without breaking single-segment text models.
DEFAULT_LAYER_PATTERN = re.compile(r"^(?:[^.]+\.)?model\.layers\.(\d+)\.(.+)$")

# Tensors that always live outside any one layer. FlatRun splits them
# into a "pre" bookend (loaded with the first layer) and a "post"
# bookend (loaded with the last layer). Both the bare ``model.``
# naming and the ``language_model.model.`` prefix used by Qwen3.5
# multimodal checkpoints are accepted.
DEFAULT_PRE_LAYER_TENSORS = (
    "model.embed_tokens.weight",
    "model.embed_tokens.biases",
    "model.embed_tokens.scales",
    "model.rotary_emb.inv_freq",
    "language_model.model.embed_tokens.weight",
    "language_model.model.embed_tokens.biases",
    "language_model.model.embed_tokens.scales",
    "language_model.model.rotary_emb.inv_freq",
)

DEFAULT_POST_LAYER_TENSORS = (
    "model.norm.weight",
    "lm_head.weight",
    "lm_head.bias",
    "lm_head.biases",
    "lm_head.scales",
    "language_model.model.norm.weight",
    "language_model.lm_head.weight",
    "language_model.lm_head.bias",
    "language_model.lm_head.biases",
    "language_model.lm_head.scales",
)


# Backwards-compatible alias. Older code refers to a single list.
DEFAULT_NON_LAYER_TENSORS = DEFAULT_PRE_LAYER_TENSORS + DEFAULT_POST_LAYER_TENSORS


@dataclass(slots=True)
class ModelManifest:
    """Description of how a model's tensors are grouped into layers.

    Attributes
    ----------
    layers : tuple[LayerDescriptor, ...]
        Ordered layer descriptors, one per decoder block.
    pre_layer : tuple[str, ...]
        Tensors loaded before layer 0 (embeddings, rotary buffers, ...).
    post_layer : tuple[str, ...]
        Tensors loaded after the final layer (output norm, LM head, ...).
    architecture : str | None
        Optional architecture tag (``"llama"``, ``"mistral"``, ...).
    """

    layers: tuple[LayerDescriptor, ...]
    pre_layer: tuple[str, ...] = ()
    post_layer: tuple[str, ...] = ()
    architecture: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def all_tensor_names(self) -> tuple[str, ...]:
        names: list[str] = []
        names.extend(self.pre_layer)
        for layer in self.layers:
            names.extend(layer.tensor_names)
        names.extend(self.post_layer)
        return tuple(names)


class ManifestBuilder:
    """Build a :class:`ModelManifest` from a backend's tensor list."""

    def __init__(
        self,
        layer_pattern: re.Pattern[str] = DEFAULT_LAYER_PATTERN,
        pre_layer_names: tuple[str, ...] = DEFAULT_PRE_LAYER_TENSORS,
        post_layer_names: tuple[str, ...] = DEFAULT_POST_LAYER_TENSORS,
    ) -> None:
        self._layer_pattern = layer_pattern
        self._pre_layer = set(pre_layer_names)
        self._post_layer = set(post_layer_names)
        # Legacy: accept the old single-list form too.
        if "model.embed_tokens.weight" in self._pre_layer:
            self._legacy_non_layer = (
                self._pre_layer | self._post_layer
            )
        else:
            self._legacy_non_layer = self._post_layer

    def build(self, backend: StorageBackend, *, architecture: str | None = None) -> ModelManifest:
        tensors = sorted({k.name for k in backend.list_tensors()})
        if not tensors:
            raise ConfigurationError("Backend exposes zero tensors; cannot build manifest")

        per_layer: dict[int, list[str]] = defaultdict(list)
        pre_layer: list[str] = []
        post_layer: list[str] = []
        unmatched: list[str] = []

        for name in tensors:
            match = self._layer_pattern.match(name)
            if match:
                idx = int(match.group(1))
                per_layer[idx].append(name)
            elif name in self._pre_layer:
                pre_layer.append(name)
            elif name in self._post_layer or name in self._legacy_non_layer:
                post_layer.append(name)
            else:
                unmatched.append(name)

        # Tensors that didn't match the layer pattern but aren't in the
        # whitelist are placed into the post-layer bookend by default.
        post_layer = post_layer + unmatched

        if not per_layer:
            raise ConfigurationError(
                f"No decoder layers matched pattern {self._layer_pattern.pattern!r}. "
                "Pass a custom ManifestBuilder if the model uses a different naming scheme."
            )

        indices = sorted(per_layer.keys())
        layers = tuple(
            LayerDescriptor(
                index=idx,
                tensor_names=tuple(sorted(per_layer[idx])),
            )
            for idx in indices
        )
        return ModelManifest(
            layers=layers,
            pre_layer=tuple(pre_layer),
            post_layer=tuple(post_layer),
            architecture=architecture,
        )


def build_manifest(
    backend: StorageBackend,
    *,
    architecture: str | None = None,
) -> ModelManifest:
    """Convenience wrapper that uses the default builder."""
    return ManifestBuilder().build(backend, architecture=architecture)


__all__ = [
    "DEFAULT_LAYER_PATTERN",
    "DEFAULT_NON_LAYER_TENSORS",
    "ManifestBuilder",
    "ModelManifest",
    "build_manifest",
]