"""Tokenizer adapter - pluggable, dependency-free BPE."""

from .bpe import (
    BPETokenizer,
    auto_load,
    load_from_gguf_metadata,
    load_from_tokenizer_json,
    load_from_vocab_merges,
)

__all__ = [
    "BPETokenizer",
    "auto_load",
    "load_from_gguf_metadata",
    "load_from_tokenizer_json",
    "load_from_vocab_merges",
]