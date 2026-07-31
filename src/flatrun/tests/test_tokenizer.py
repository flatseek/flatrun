"""Tests for the BPE tokenizer adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flatrun.tokenizer.bpe import (
    BPETokenizer,
    auto_load,
    load_from_tokenizer_json,
    load_from_vocab_merges,
)


# ---------------------------------------------------------------------------
# Direct BPETokenizer usage
# ---------------------------------------------------------------------------


def _toy_tokenizer() -> BPETokenizer:
    """A minimal 4-token toy vocab with a couple of merges."""
    vocab = {
        "a": 0,
        "b": 1,
        "c": 2,
        "ab": 3,
    }
    return BPETokenizer(vocab=vocab, merges=[])


def test_bpe_encode_decode_roundtrip_ascii() -> None:
    """Without merges, each char is its own token."""
    tok = _toy_tokenizer()
    ids = tok.encode("ab")
    assert ids == [0, 1]
    assert tok.decode(ids) == "ab"


def test_bpe_unicode_decode_safe() -> None:
    """Unknown chars raise KeyError when unk_token is unset."""
    tok = BPETokenizer(vocab={"hello": 0}, merges=[])
    with pytest.raises(KeyError):
        tok.encode("x")


def test_bpe_added_tokens_take_priority() -> None:
    """Added tokens are matched before the pretokeniser."""
    tok = BPETokenizer(
        vocab={"hello": 0, "world": 1},
        merges=[],
        added_tokens={2: "<|special|>"},
    )
    ids = tok.encode("<|special|>")
    assert ids == [2]
    assert tok.decode([2]) == "<|special|>"


def test_bpe_byte_level_maps_high_bytes() -> None:
    """Non-ASCII bytes map to their GPT-2 byte-level equivalents."""
    from flatrun.tokenizer.bpe import _bytes_to_unicode
    bs = _bytes_to_unicode()
    raw_bytes = "é".encode("utf-8")
    tok = BPETokenizer(
        vocab={
            bs[int(raw_bytes[0])]: 10,
            bs[int(raw_bytes[1])]: 11,
        },
        merges=[],
    )
    ids = tok.encode("é")
    assert 10 in ids
    assert 11 in ids


def test_bpe_with_unk_token_falls_back() -> None:
    """Unknown chars map to ``unk_token`` when one is configured."""
    tok = BPETokenizer(vocab={"hello": 0, "<unk>": 99}, merges=[], unk_token="<unk>")
    ids = tok.encode("x")
    assert ids == [99]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def test_load_from_vocab_merges(tmp_path: Path) -> None:
    vocab = {"hello": 0, "world": 1}
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab))
    merges_path = tmp_path / "merges.txt"
    merges_path.write_text("# top comment\nh e\n")
    tok = load_from_vocab_merges(vocab_path, merges_path)
    assert tok.vocab == vocab
    assert tok.merges == [("h", "e")]


def test_load_from_vocab_merges_with_added_tokens(tmp_path: Path) -> None:
    vocab = {"x": 0}
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab))
    merges_path = tmp_path / "merges.txt"
    merges_path.write_text("")
    added = tmp_path / "added_tokens.json"
    added.write_text(json.dumps({"100": {"content": "<|pad|>"}}))
    tok = load_from_vocab_merges(vocab_path, merges_path, added_tokens_path=added)
    assert tok.added_tokens[100] == "<|pad|>"


def test_load_from_vocab_merges_other_layout(tmp_path: Path) -> None:
    """``added_tokens.json`` may also be ``{token: id}``."""
    vocab = {"x": 0}
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab))
    merges_path = tmp_path / "merges.txt"
    merges_path.write_text("")
    added = tmp_path / "added_tokens.json"
    added.write_text(json.dumps({"<eos>": 200}))
    tok = load_from_vocab_merges(vocab_path, merges_path, added_tokens_path=added)
    assert tok.added_tokens[200] == "<eos>"


def test_load_from_tokenizer_json(tmp_path: Path) -> None:
    data = {
        "model": {
            "type": "BPE",
            "vocab": {"hello": 0, "world": 1},
            "merges": ["h e", "he llo"],
        },
        "added_tokens": [{"id": 2, "content": "<eos>"}],
    }
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(data))
    tok = load_from_tokenizer_json(path)
    assert tok.vocab["hello"] == 0
    assert tok.merges == [("h", "e"), ("he", "llo")]
    assert tok.added_tokens[2] == "<eos>"


def test_auto_load_prefers_tokenizer_json(tmp_path: Path) -> None:
    """When both formats exist, ``tokenizer.json`` wins."""
    data = {"model": {"type": "BPE", "vocab": {"auto": 0}, "merges": []}}
    (tmp_path / "tokenizer.json").write_text(json.dumps(data))
    (tmp_path / "vocab.json").write_text(json.dumps({"other": 5}))
    tok = auto_load(tmp_path)
    assert "auto" in tok.vocab
    assert "other" not in tok.vocab


def test_auto_load_falls_back_to_vocab_merges(tmp_path: Path) -> None:
    (tmp_path / "vocab.json").write_text(json.dumps({"x": 0}))
    (tmp_path / "merges.txt").write_text("")
    tok = auto_load(tmp_path)
    assert tok.vocab == {"x": 0}


def test_auto_load_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        auto_load(tmp_path)


# ---------------------------------------------------------------------------
# Real-world smoke test against the user's LM Studio download
# ---------------------------------------------------------------------------


def test_real_qwen7b_tokenizer_loads() -> None:
    """Round-trip a real string through the production tokenizer."""
    model_dir = Path(
        "/Users/judotens/.lmstudio/models/lmstudio-community/"
        "Qwen2.5-Coder-7B-Instruct-MLX-4bit"
    )
    if not (model_dir / "tokenizer.json").is_file():
        pytest.skip(f"tokenizer.json not found at {model_dir}")
    tok = auto_load(model_dir)
    assert len(tok.vocab) > 1000
    ids = tok.encode("def hello():")
    assert len(ids) == 3
    assert tok.decode(ids).strip() == "def hello():"



# ---------------------------------------------------------------------------
# Chat templates
# ---------------------------------------------------------------------------


def test_apply_chat_template_default_qwen2_chatml() -> None:
    """Default template renders Qwen2 ChatML with the assistant prompt opener."""
    tok = BPETokenizer(vocab={"<|im_start|>": 0, "<|im_end|>": 1}, merges=[])
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
    ]
    out = tok.apply_chat_template(messages, add_generation_prompt=True)
    assert "<|im_start|>system\nBe helpful.<|im_end|>" in out
    assert "<|im_start|>user\nHi<|im_end|>" in out
    assert out.endswith("<|im_start|>assistant\n")


def test_apply_chat_template_without_generation_prompt() -> None:
    tok = BPETokenizer(vocab={}, merges=[])
    out = tok.apply_chat_template(
        [{"role": "user", "content": "x"}], add_generation_prompt=False
    )
    assert "<|im_start|>assistant" not in out
    assert out.strip() == "<|im_start|>user\nx<|im_end|>"


def test_apply_chat_template_handles_if_endif() -> None:
    """Custom Jinja template with {%- if %} branch."""
    template = (
        "{%- for m in messages %}"
        "{{- '[' + m['role'] + ']' + m['content'] }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '[GEN]' }}"
        "{%- endif %}"
    )
    tok = BPETokenizer(vocab={}, merges=[], chat_template=template)
    out = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True
    )
    assert out == "[user]hi[GEN]"


def test_apply_chat_template_roundtrip_with_added_tokens() -> None:
    """Special tokens in the template encode as their added IDs."""
    # Provide a vocab with the special tokens plus every byte-level char
    # the prompt could split into.
    from flatrun.tokenizer.bpe import _bytes_to_unicode
    bs = _bytes_to_unicode()
    base = {"<|im_start|>": 0, "<|im_end|>": 1}
    # Add the byte-level form of every character that appears in the
    # rendered prompt: 'u', 's', 'e', 'r', '\n'.
    for c in "user\n":
        base[bs[ord(c)]] = 100 + ord(c)
    tok = BPETokenizer(vocab=base, merges=[], added_tokens={0: "<|im_start|>", 1: "<|im_end|>"})
    messages = [{"role": "user", "content": ""}]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=False)
    ids = tok.encode(prompt)
    # The prompt is "<|im_start|>user\\n<|im_end|>\\n" so the special
    # token IDs (0, 1) appear in order with byte-level fragments between.
    assert ids[0] == 0
    assert 1 in ids


def test_chat_template_loaded_from_tokenizer_config(tmp_path: Path) -> None:
    """A tokenizer_config.json with chat_template is honoured by auto_load."""
    # Set up a tiny tokenizer.json next to a tokenizer_config.json.
    tok_data = {
        "model": {"type": "BPE", "vocab": {"hello": 0}, "merges": []},
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok_data))
    cfg = {
        "chat_template": "{%- for m in messages %}{{ m['content'] }}{%- endfor %}",
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg))
    tok = auto_load(tmp_path)
    out = tok.apply_chat_template(
        [{"role": "user", "content": "HELLO"}], add_generation_prompt=False
    )
    assert out == "HELLO"


def test_chat_template_falls_back_when_config_missing() -> None:
    """No tokenizer_config.json -> Qwen2 default template."""
    tok = BPETokenizer(vocab={"<|im_start|>": 0, "<|im_end|>": 1}, merges=[])
    # chat_template is the Qwen2 default after construction.
    assert "<|im_start|>" in tok.chat_template



# ---------------------------------------------------------------------------
# GGUF-backed tokenizer
# ---------------------------------------------------------------------------


def test_load_from_gguf_metadata_returns_full_vocab(tmp_path) -> None:
    """A stub GGUF backend hands the loader a vocab + merges + specials."""
    from flatrun.tokenizer.bpe import load_from_gguf_metadata

    # Hand-craft a minimal metadata dict. We mock the backend by
    # subclassing the import path through a local stub.
    import sys
    import types

    class _StubBackend:
        def __init__(self, path) -> None:
            self.path = path

        def open(self) -> None:
            pass

        def close(self) -> None:
            pass

        @property
        def gguf_metadata(self) -> dict:
            return {
                "tokenizer.ggml.tokens": [
                    "a", "b", "c", "<|endoftext|>", "<|im_start|>", "<|im_end|>"
                ],
                "tokenizer.ggml.token_type": [1, 1, 1, 3, 3, 3],
                "tokenizer.ggml.merges": ["a b", "b c"],
                "tokenizer.ggml.padding_token_id": 0,
                "tokenizer.ggml.add_bos_token": False,
            }

    stub_module = types.ModuleType("stub_gguf")
    stub_module.GGUFBackend = _StubBackend
    sys.modules["stub_gguf"] = stub_module

    tok = load_from_gguf_metadata(
        tmp_path / "fake.gguf",
        backend_module="stub_gguf",
        backend_class="GGUFBackend",
    )
    assert len(tok.vocab) == 6
    assert tok.vocab["<|im_start|>"] == 4
    # Special tokens (token_type 3) are added to the added_tokens table.
    assert 3 in tok.added_tokens
    assert tok.added_tokens[3] == "<|endoftext|>"
    assert tok.merges == [("a", "b"), ("b", "c")]
    # Unk token is set to the padding-token-id.
    assert tok.unk_token == "a"


def test_load_from_gguf_metadata_loads_chat_template(tmp_path) -> None:
    import sys
    import types

    class _StubBackend:
        def __init__(self, path) -> None:
            pass

        def open(self) -> None:
            pass

        def close(self) -> None:
            pass

        @property
        def gguf_metadata(self) -> dict:
            return {
                "tokenizer.ggml.tokens": ["a", "b"],
                "tokenizer.ggml.token_type": [1, 1],
                "tokenizer.ggml.merges": [],
                "tokenizer.chat_template": "{%- for m in messages %}{{ m['content'] }}{%- endfor %}",
            }

    stub_module = types.ModuleType("stub_gguf2")
    stub_module.GGUFBackend = _StubBackend
    sys.modules["stub_gguf2"] = stub_module

    from flatrun.tokenizer.bpe import load_from_gguf_metadata
    tok = load_from_gguf_metadata(
        tmp_path / "fake.gguf",
        backend_module="stub_gguf2",
        backend_class="GGUFBackend",
    )
    assert "for m in messages" in tok.chat_template


def test_auto_load_falls_back_to_gguf(tmp_path) -> None:
    """A directory with only a .gguf file uses load_from_gguf_metadata."""
    import sys
    import types

    class _StubBackend:
        def __init__(self, path) -> None:
            pass

        def open(self) -> None:
            pass

        def close(self) -> None:
            pass

        @property
        def gguf_metadata(self) -> dict:
            return {
                "tokenizer.ggml.tokens": ["x", "y", "z"],
                "tokenizer.ggml.token_type": [1, 1, 1],
                "tokenizer.ggml.merges": [],
            }

    stub_module = types.ModuleType("stub_gguf3")
    stub_module.GGUFBackend = _StubBackend
    sys.modules["stub_gguf3"] = stub_module

    # Patch the loader used by auto_load to use our stub backend.
    import flatrun.tokenizer.bpe as bpe
    orig = bpe.load_from_gguf_metadata

    def patched(path):
        return orig(
            path,
            backend_module="stub_gguf3",
            backend_class="GGUFBackend",
        )

    bpe.load_from_gguf_metadata = patched
    try:
        gguf_file = tmp_path / "fake.gguf"
        gguf_file.write_bytes(b"GGUF" + b"\x00" * 100)
        tok = bpe.auto_load(tmp_path)
        assert tok.vocab == {"x": 0, "y": 1, "z": 2}
    finally:
        bpe.load_from_gguf_metadata = orig
