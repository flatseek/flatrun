"""Minimal GPT-2 / Qwen2 style BPE tokenizer.

Pure-Python implementation that reads the HuggingFace tokenizer
artifacts and produces token IDs without any third-party dependency.
The encoder follows the algorithm described in
https://huggingface.co/docs/transformers/tokenizer_summary.

Limitations vs. the upstream ``tokenizers`` library:

* No pretokenisation regex tweaks per model (uses a simple GPT-2 style
  pre-tokeniser that handles ``[a-zA-Z]+|[0-9]+|[^a-zA-Z0-9]+``).
* No byte-level fallback for non-UTF-8 bytes (we treat input as UTF-8).
* Slow - the encoder is O(N * log N) per word; the fast Rust encoder
  does it in linear time. This is fine for prompts up to a few KiB.

What we DO support:

* Added special tokens (via ``added_tokens.json`` or the ``added_tokens``
  block in ``tokenizer.json``).
* Byte-level BPE (GPT-2 style 256-byte alphabet).
* SentencePiece-style BPE (Qwen2, LLaMA) by reading vocab.json + merges.txt.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable


# GPT-2 style byte-level encoder. Maps each byte to a printable Unicode
# character so it can be a token in a JSON vocab.
@lru_cache(maxsize=None)
def _bytes_to_unicode() -> dict[int, str]:
    """Return the byte-to-unicode mapping used by GPT-2/Qwen BPE.

    The mapping covers all 256 byte values; printable ASCII bytes map
    to themselves, the rest map into the '!'..'~' + '¡'..'¬' + '®'..'ÿ'
    range.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Return the set of adjacent symbol pairs in ``word``."""
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


# Simple pretokeniser: letters, digits, punctuation, whitespace.
_PRETOK_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


def _gpt2_pretok(text: str) -> list[str]:
    """Pretokenise ``text`` into a list of word fragments.

    This is a simplified version of the GPT-2 pretokeniser that still
    produces the right fragments for English text. It is not byte-for-
    byte identical to the upstream implementation.
    """
    return _PRETOK_PATTERN.findall(text)


class BPETokenizer:
    """GPT-2 / Qwen2 style BPE tokenizer.

    Parameters
    ----------
    vocab : dict[str, int]
        Token string -> id.
    merges : list[tuple[str, str]]
        BPE merge rules in priority order.
    added_tokens : dict[int, str]
        Special / added tokens keyed by id.
    unk_token : str | None
        Unknown token. When ``None`` unknown bytes raise.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        added_tokens: dict[int, str] | None = None,
        unk_token: str | None = "<|endoftext|>",
        chat_template: str | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        # Inverse vocab: id -> token string.
        self.inv_vocab: dict[int, str] = {i: t for t, i in vocab.items()}
        # Added / special tokens - highest priority in encoder.
        self.added_tokens = dict(added_tokens or {})
        self.unk_token = unk_token
        # Explicit end-of-sequence token id, when declared by the model
        # (GGUF ``tokenizer.ggml.eos_token_id`` or ``tokenizer_config.json``).
        self.eos_token_id = eos_token_id

        # Pre-compute BPE merge ranks (lower = applied first).
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        # Build a regex that matches any added token literally so we
        # can escape them from the pretokeniser.
        self.added_token_regex: re.Pattern[str] | None = None
        if self.added_tokens:
            # Sort by length descending so longer tokens match first.
            tokens = sorted(self.added_tokens.values(), key=len, reverse=True)
            escaped = [re.escape(t) for t in tokens if t]
            if escaped:
                self.added_token_regex = re.compile("|".join(escaped))

        # Chat template - Jinja-style. Defaults to Qwen2 ChatML.
        self.chat_template = chat_template or DEFAULT_QWEN2_CHAT_TEMPLATE

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` to a list of token IDs."""
        bpe_tokens: list[int] = []
        # Split text into chunks: added tokens and pretokenised fragments.
        chunks = self._split_for_added_tokens(text)
        for chunk, is_added in chunks:
            if is_added:
                bpe_tokens.append(self._token_to_id(chunk))
                continue
            for word in _gpt2_pretok(chunk):
                bpe_tokens.extend(self._encode_word(word))
        return bpe_tokens

    def _split_for_added_tokens(self, text: str) -> list[tuple[str, bool]]:
        """Walk ``text`` and yield ``(chunk, is_added)`` tuples.

        Added tokens (longest match first) split the text; everything
        between them goes through the normal pretokeniser.
        """
        if self.added_token_regex is None:
            return [(text, False)]
        result: list[tuple[str, bool]] = []
        last = 0
        for match in self.added_token_regex.finditer(text):
            if match.start() > last:
                result.append((text[last : match.start()], False))
            result.append((match.group(0), True))
            last = match.end()
        if last < len(text):
            result.append((text[last:], False))
        return result

    def _encode_word(self, word: str) -> list[int]:
        """Encode one pretokenised word fragment to token IDs."""
        # Byte-level encoding.
        bpe_input = "".join(_bytes_to_unicode()[b] for b in word.encode("utf-8"))
        if not bpe_input:
            return []
        bpe_result = self._bpe(bpe_input)
        return [self._token_to_id(t) for t in bpe_result]

    def _token_to_id(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        # Try added tokens.
        for tid, t in self.added_tokens.items():
            if t == token:
                return tid
        if self.unk_token is not None and self.unk_token in self.vocab:
            return self.vocab[self.unk_token]
        raise KeyError(f"Unknown token: {token!r}")

    def _bpe(self, word: str) -> list[str]:
        """Apply BPE merges to ``word`` (a byte-level encoded string)."""
        if not word:
            return []
        word = tuple(word)  # type: ignore[assignment]
        if len(word) == 1:
            return list(word)
        pairs = _get_pairs(word)
        if not pairs:
            return list(word)
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        return list(word)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: Iterable[int]) -> str:
        """Decode a list of token IDs back to a string."""
        pieces: list[str] = []
        for tid in ids:
            # Added tokens override the vocab.
            if tid in self.added_tokens:
                pieces.append(self.added_tokens[tid])
                continue
            tok = self.inv_vocab.get(tid)
            if tok is None:
                continue
            pieces.append(tok)
        text = "".join(pieces)
        # Reverse byte-level encoding.
        byte_map = {v: k for k, v in _bytes_to_unicode().items()}
        raw = bytearray()
        for ch in text:
            if ch in byte_map:
                raw.append(byte_map[ch])
            else:
                # Not a byte-level token; emit UTF-8 of the char.
                raw.extend(ch.encode("utf-8", errors="replace"))
        return raw.decode("utf-8", errors="replace")

    def stop_token_ids(self) -> set[int]:
        """Return the set of token ids that should end generation.

        Prefers the model's explicitly-declared ``eos_token_id`` (read
        from GGUF ``tokenizer.ggml.eos_token_id`` or ``tokenizer_config.json``).
        When the model does not declare one, falls back to scanning the
        added/special-token table for common end-of-turn markers
        (``<|endoftext|>``, ``<|im_end|>``, ``</s>``, ``<|end|>``,
        ``<|eot_id|>``).
        """
        if self.eos_token_id is not None:
            return {int(self.eos_token_id)}
        stop: set[int] = set()
        for tid, tok in self.added_tokens.items():
            if any(s in tok for s in ("im_end", "endoftext", "/s>", "end>", "eot_id")):
                stop.add(int(tid))
        return stop

    # ------------------------------------------------------------------
    # Chat templates
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render ``messages`` to a prompt string using the chat template.

        Parameters
        ----------
        messages : list of dict
            Each message is ``{"role": ..., "content": ...}``. Roles are
            typically ``"system"``, ``"user"``, ``"assistant"``.
        add_generation_prompt : bool
            When ``True`` (default) append the assistant turn opener so
            the model can continue with its reply. Set ``False`` for
            training-style continuations.

        Returns
        -------
        str
            The rendered prompt. Feed it to :meth:`encode` to get IDs.

        Notes
        -----
        The default template is Qwen2 ChatML. If a more complex template
        is needed (e.g. tool calls, multi-modal content) pass a Jinja
        string to the constructor's ``chat_template`` argument. The
        renderer is a small dependency-free subset of Jinja: see
        :func:`_render_jinja_chat` for what it supports.

        When the loaded template uses features our renderer doesn't
        support (e.g. ``loop.index0``, tool calls), the call falls back
        to a simple ChatML renderer that ignores those features.
        """
        try:
            return _render_jinja_chat(
                self.chat_template,
                messages,
                add_generation_prompt=add_generation_prompt,
            )
        except (ValueError, SyntaxError, NameError, TypeError):
            # Fall back to a ChatML-only renderer. We still honour
            # ``add_generation_prompt`` so the caller gets the
            # assistant opener when requested.
            return _format_qwen2_messages(messages, add_generation_prompt)


def _format_qwen2_messages(messages: list[dict], add_generation_prompt: bool) -> str:
    """Render a simple Qwen2 ChatML prompt from a list of messages.

    Used as the fallback when a model's ``tokenizer_config.json``
    ships a Jinja template that uses features our tiny renderer
    doesn't support (e.g. ``loop.index0``, tool calls, attribute
    access on list elements). The output is byte-compatible with
    the Qwen2 ChatML format.
    """
    out: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
    return "".join(out)


# ---------------------------------------------------------------------------
# Chat templates
# ---------------------------------------------------------------------------


# Default Qwen2 / Qwen2.5 ChatML template. This is what HuggingFace's
# Qwen2 family uses out of the box. Users can override per-tokenizer by
# passing ``chat_template=<jinja string>`` to the constructor.
DEFAULT_QWEN2_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


# --- Tiny Jinja subset (no external dep) --------------------------------
# Supports: {{ expr }}, {%- if EXPR %}{%- else %}{%- endif %},
# {%- for VAR in EXPR %}{%- endfor %}. Expressions are evaluated as
# Python with a restricted namespace.

_TAG_RE = re.compile(r"\{\%\s*-?\s*(if|else|endif|for|endfor)\s*(.*?)\s*-?\s*%\}")
_EXPR_RE = re.compile(r"\{\{\s*-?\s*(.*?)\s*-?\s*\}\}")


def _eval_expr(expr: str, scope: dict) -> object:
    """Evaluate a Python expression with a restricted namespace."""
    safe = {**scope, "True": True, "False": False, "None": None}
    try:
        return eval(expr, {"__builtins__": {}}, safe)
    except Exception:
        return ""


def _find_matching(template: str, start: int, open_kind: str, close_kind: str) -> int:
    """Find the index of the matching close tag, ignoring nested same-kind pairs.

    Skips over ``{{ ... }}`` expression blocks so that ``{%- for ... %}``
    inside an interpolated expression does not affect the depth counter.
    """
    depth = 1
    i = start
    while i < len(template):
        # Skip over expression blocks first - they may contain literal
        # ``{%`` characters that we don't want to count as flow control.
        e = _EXPR_RE.match(template, i)
        if e:
            i = e.end()
            continue
        m = _TAG_RE.match(template, i)
        if not m:
            break
        if m.group(1) == open_kind:
            depth += 1
        elif m.group(1) == close_kind:
            depth -= 1
            if depth == 0:
                return m.start()
        i = m.end()
    return -1


def _render_jinja_chat(template: str, messages: list, *, add_generation_prompt: bool) -> str:
    """Render a Jinja chat template against a message list.

    Handles ``{{ expr }}`` interpolation and ``{%- if/for %}{%- endif/endfor %}``
    flow control. ``messages`` may be a list of dicts (each with ``role``
    and ``content`` keys). The renderer is intentionally tiny - it does
    not support filters, macros, or includes.
    """
    scope = {
        "messages": messages,
        "add_generation_prompt": bool(add_generation_prompt),
    }
    return _process_segment(template, scope)


def _process_segment(text: str, scope: dict) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        m_tag = _TAG_RE.match(text, i)
        m_expr = _EXPR_RE.match(text, i)
        if m_tag:
            kind = m_tag.group(1)
            expr = m_tag.group(2).strip()
            tag_end = m_tag.end()
            if kind == "if":
                cond = bool(_eval_expr(expr, scope))
                # Locate matching endif (and any else within).
                depth = 1
                j = tag_end
                else_pos = -1
                end_pos = -1
                while j < len(text):
                    # Skip past expression blocks so they don't shift depth.
                    ee = _EXPR_RE.match(text, j)
                    if ee:
                        j = ee.end()
                        continue
                    mm = _TAG_RE.match(text, j)
                    if not mm:
                        break
                    if mm.group(1) == "if":
                        depth += 1
                    elif mm.group(1) == "endif":
                        depth -= 1
                        if depth == 0:
                            end_pos = mm.start()
                            break
                    elif mm.group(1) == "else" and depth == 1:
                        else_pos = mm.start()
                    j = mm.end()
                if end_pos < 0:
                    raise ValueError("if without endif")
                end_match = _TAG_RE.match(text, end_pos)
                if else_pos < 0:
                    body = text[tag_end:end_pos]
                    if cond:
                        out.append(_process_segment(body, scope))
                else:
                    else_match = _TAG_RE.match(text, else_pos)
                    if_body = text[tag_end:else_pos]
                    else_body = text[else_match.end():end_pos]
                    if cond:
                        out.append(_process_segment(if_body, scope))
                    else:
                        out.append(_process_segment(else_body, scope))
                i = end_match.end()
                continue
            if kind == "for":
                m_for = re.match(r"(\w+)\s+in\s+(\S+)", expr)
                if not m_for:
                    raise ValueError(f"Unsupported for-expression: {expr!r}")
                var_name, seq_expr = m_for.group(1), m_for.group(2)
                end_pos = _find_matching(text, tag_end, "for", "endfor")
                if end_pos < 0:
                    raise ValueError("for without endfor")
                end_match = _TAG_RE.match(text, end_pos)
                body = text[tag_end:end_pos]
                seq = _eval_expr(seq_expr, scope) or []
                for item in seq:
                    out.append(_process_segment(body, {**scope, var_name: item}))
                i = end_match.end()
                continue
            # endif/endfor/else are not expected at top level.
            i = tag_end
            continue
        if m_expr:
            out.append(str(_eval_expr(m_expr.group(1).strip(), scope)))
            i = m_expr.end()
            continue
        out.append(text[i])
        i += 1
    return "".join(out)



def _read_merges(path: Path) -> list[tuple[str, str]]:
    merges: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
    return merges


def load_from_vocab_merges(
    vocab_path: Path | str,
    merges_path: Path | str,
    added_tokens_path: Path | str | None = None,
    tokenizer_config_path: Path | str | None = None,
) -> BPETokenizer:
    """Load a tokenizer from vocab.json + merges.txt + added_tokens.json.

    This is the layout shipped with Qwen2, LLaMA, and most HuggingFace
    repositories that don't rely on the fast tokenizer.

    If ``tokenizer_config_path`` points to a ``tokenizer_config.json``
    file, any ``chat_template`` it contains is forwarded to the
    :class:`BPETokenizer` so :meth:`BPETokenizer.apply_chat_template`
    uses the model author's template instead of the Qwen2 default.
    """
    vocab_path = Path(vocab_path)
    merges_path = Path(merges_path)
    with open(vocab_path, "r", encoding="utf-8") as fh:
        vocab_raw = json.load(fh)
    vocab = {k: int(v) for k, v in vocab_raw.items()}
    merges = _read_merges(merges_path)

    added_tokens: dict[int, str] = {}
    if added_tokens_path is not None and Path(added_tokens_path).is_file():
        with open(added_tokens_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Two layouts:
        # - {"151643": {"content": "<|endoftext|>", ...}, ...}
        # - {"<|endoftext|>": 151643, ...}
        for k, v in data.items():
            if isinstance(v, dict):
                tid = int(k)
                content = str(v.get("content", ""))
                added_tokens[tid] = content
            else:
                added_tokens[int(v)] = str(k)

    chat_template = _read_chat_template(tokenizer_config_path)
    return BPETokenizer(
        vocab=vocab,
        merges=merges,
        added_tokens=added_tokens,
        chat_template=chat_template,
        eos_token_id=_read_eos_token_id(tokenizer_config_path, vocab, added_tokens),
    )


def load_from_tokenizer_json(
    path: Path | str,
    tokenizer_config_path: Path | str | None = None,
) -> BPETokenizer:
    """Load a tokenizer from the modern HuggingFace ``tokenizer.json`` format.

    Supports:

    * WordLevel + BPE backends,
    * Added tokens at the top level,
    * The standard byte-level pretokeniser.

    If ``tokenizer_config_path`` points to a ``tokenizer_config.json``
    file, any ``chat_template`` it contains is forwarded to the
    :class:`BPETokenizer` so :meth:`BPETokenizer.apply_chat_template`
    uses the model author's template instead of the Qwen2 default.

    Returns a :class:`BPETokenizer` ready for use.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    model = data.get("model", {})
    vocab_type = model.get("type", "BPE")
    if vocab_type not in ("BPE", "WordLevel", "WordPiece"):
        raise ValueError(f"Tokenizer model type {vocab_type!r} not supported")

    # Build vocab.
    vocab: dict[str, int] = {}
    for tok, idx in model.get("vocab", {}).items():
        vocab[tok] = int(idx)

    # Collect merges. In HuggingFace format, each merge is a pair of
    # strings; the upstream encoder stores them under "merges".
    merges_raw = model.get("merges", [])
    merges: list[tuple[str, str]] = []
    for entry in merges_raw:
        if isinstance(entry, str):
            parts = entry.split(" ", 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
        elif isinstance(entry, list) and len(entry) == 2:
            merges.append((str(entry[0]), str(entry[1])))

    added_tokens: dict[int, str] = {}
    for tok in data.get("added_tokens", []):
        tid = int(tok.get("id", -1))
        content = str(tok.get("content", ""))
        if tid >= 0 and content:
            added_tokens[tid] = content

    chat_template = _read_chat_template(tokenizer_config_path)
    return BPETokenizer(
        vocab=vocab,
        merges=merges,
        added_tokens=added_tokens,
        chat_template=chat_template,
        eos_token_id=_read_eos_token_id(tokenizer_config_path, vocab, added_tokens),
    )


def _read_chat_template(path: Path | str | None) -> str | None:
    """Read a ``chat_template`` field from a ``tokenizer_config.json``.

    Returns ``None`` if the file doesn't exist, can't be parsed, or
    doesn't contain a chat template - the caller falls back to the
    Qwen2 default.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    template = data.get("chat_template")
    if not isinstance(template, str) or not template.strip():
        return None
    return template


def _read_eos_token_id(
    path: Path | str | None,
    vocab: dict[str, int] | None = None,
    added_tokens: dict[int, str] | None = None,
) -> int | None:
    """Read an ``eos_token_id`` from a ``tokenizer_config.json``.

    Prefers the explicit ``eos_token_id`` integer field (HF layout).
    Falls back to resolving the ``eos_token`` string/object through the
    vocab / added-token tables (the layout Flatbuild writes).
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    tid = data.get("eos_token_id")
    if isinstance(tid, int):
        return int(tid)

    eos = data.get("eos_token")
    if isinstance(eos, dict):
        eos = eos.get("content")
    if isinstance(eos, str):
        if vocab is not None and eos in vocab:
            return int(vocab[eos])
        if added_tokens is not None:
            for added_id, token in added_tokens.items():
                if token == eos:
                    return int(added_id)
    return None


# GGUF ``tokenizer.ggml.*`` type ids used to distinguish normal vs
# special tokens. The values match llama.cpp / GGML conventions.
_GGML_TOKEN_TYPE_NORMAL = 1
_GGML_TOKEN_TYPE_CONTROL = 2   # All Up and Empty token style (unused here)
_GGML_TOKEN_TYPE_SPECIAL = 3   # Added / special tokens
_GGML_TOKEN_TYPE_USERDEF = 4   # Tool-call style markers


def load_from_gguf_metadata(
    gguf_path: Path | str,
    *,
    backend_module: str = "flatrun.backend.gguf",
    backend_class: str = "GGUFBackend",
) -> BPETokenizer:
    """Build a :class:`BPETokenizer` from a GGUF file's metadata KV table.

    GGUF files embed the full BPE vocabulary + merges + special tokens
    under the ``tokenizer.ggml.*`` keys, so models that ship only the
    ``.gguf`` (no ``tokenizer.json``) can still be tokenised correctly.

    The chat template, if present in the GGUF metadata, is forwarded
    to the :class:`BPETokenizer`.

    Parameters
    ----------
    gguf_path : Path or str
        Path to the ``.gguf`` file.
    backend_module / backend_class : str
        Override the import path of the backend. Tests can stub a
        fake metadata source by passing a custom module + class.
    """
    import importlib
    module = importlib.import_module(backend_module)
    backend = getattr(module, backend_class)(gguf_path)
    backend.open()
    try:
        meta = backend.gguf_metadata
    finally:
        backend.close()

    tokens = meta.get("tokenizer.ggml.tokens") or []
    if not tokens:
        raise ValueError(
            f"GGUF file {gguf_path} has no tokenizer.ggml.tokens list"
        )

    # Build vocab: token string -> id.
    vocab: dict[str, int] = {tok: i for i, tok in enumerate(tokens)}

    # Collect special / added tokens. token_type lists are aligned with
    # the tokens list - index i corresponds to tokens[i]. Anything
    # flagged as a special type is recorded as an added token.
    token_types = meta.get("tokenizer.ggml.token_type") or []
    added_tokens: dict[int, str] = {}
    for i, ty in enumerate(token_types):
        if ty in (_GGML_TOKEN_TYPE_SPECIAL, _GGML_TOKEN_TYPE_USERDEF) and i < len(tokens):
            added_tokens[i] = tokens[i]

    # Convert the merge strings ("a b" -> ("a", "b")) into tuples.
    raw_merges = meta.get("tokenizer.ggml.merges") or []
    merges: list[tuple[str, str]] = []
    for entry in raw_merges:
        parts = entry.split(" ", 1)
        if len(parts) == 2:
            merges.append((parts[0], parts[1]))

    # The GGUF metadata's chat template (if any) is preferred over the
    # Qwen2 default. The Qwen2 default also lives in ``tokenizer.ggml.pre``
    # but the explicit chat_template field is what GGUF writes for
    # instruction-tuned models.
    chat_template = meta.get("tokenizer.chat_template")
    if not isinstance(chat_template, str) or not chat_template.strip():
        chat_template = None

    # Pick an ``unk_token``. GGUF exposes ``padding_token_id`` which is
    # the closest equivalent to a benign fallback id.
    unk_id = int(meta.get("tokenizer.ggml.padding_token_id", 0))
    unk_token = tokens[unk_id] if 0 <= unk_id < len(tokens) else None

    # Explicit end-of-sequence id, when the GGUF declares one.
    eos_raw = meta.get("tokenizer.ggml.eos_token_id")
    eos_token_id = int(eos_raw) if isinstance(eos_raw, int) else None

    return BPETokenizer(
        vocab=vocab,
        merges=merges,
        added_tokens=added_tokens,
        unk_token=unk_token,
        chat_template=chat_template,
        eos_token_id=eos_token_id,
    )


def auto_load(model_dir: Path | str) -> BPETokenizer:
    """Pick the best tokenizer files available in ``model_dir``."""
    model_dir = Path(model_dir)
    tj = model_dir / "tokenizer.json"
    if tj.is_file():
        return load_from_tokenizer_json(
            tj, tokenizer_config_path=model_dir / "tokenizer_config.json"
        )
    vocab = model_dir / "vocab.json"
    merges = model_dir / "merges.txt"
    if vocab.is_file() and merges.is_file():
        return load_from_vocab_merges(
            vocab,
            merges,
            added_tokens_path=model_dir / "added_tokens.json",
            tokenizer_config_path=model_dir / "tokenizer_config.json",
        )
    # GGUF-only directory: build the tokenizer from the .gguf file.
    gguf_files = sorted(model_dir.glob("*.gguf"))
    if gguf_files:
        return load_from_gguf_metadata(gguf_files[0])
    raise FileNotFoundError(
        f"No tokenizer files found in {model_dir} (looked for tokenizer.json, "
        "vocab.json + merges.txt, or .gguf)"
    )


__all__ = [
    "BPETokenizer",
    "DEFAULT_QWEN2_CHAT_TEMPLATE",
    "auto_load",
    "load_from_gguf_metadata",
    "load_from_tokenizer_json",
    "load_from_vocab_merges",
]