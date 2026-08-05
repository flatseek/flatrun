"""Generation engine — wraps a loaded model bundle for HTTP serving.

A :class:`GenerationEngine` owns the same state the CLI ``run`` /
``chat`` commands build:

* the loaded model (``runtime``, ``manifest``, ``config``)
* a tokenizer (auto-detected from GGUF metadata or a sibling dir)
* a :class:`StreamingExecutor` with a preallocated KV cache
* a :class:`Sampler`

It exposes a streaming API (:meth:`stream_chat`,
:meth:`stream_complete`) that yields one decoded token string at a
time. Callers (``openai.py``, ``anthropic.py``) are responsible for
shaping those tokens into the appropriate SSE wire format.

Why a class and not free functions? Each server process holds
exactly one model — splitting state into a class makes the lock and
the per-request ``reset()`` explicit, and avoids accidental global
state in the FastAPI routes. ``acquire()`` is the concurrency
primitive: a single request runs through to completion because
streaming models can't safely be interleaved on the same executor
(the scheduler resets the KV cache per call). When async fan-out
arrives, this is the place to add a queue.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .. import KVCache, RuntimeConfig, StreamingExecutor, load_huggingface
from ..cli import _build_config_from_gguf, _pick_gguf_file, _resolve_model_paths
from ..model.qwen2 import Qwen2Config, make_qwen2_forwarder
from ..model.sampling import Sampler
from ..runtime.backend import get_backend
from ..runtime.memory import MemoryConfig
from ..tokenizer import auto_load, load_from_gguf_metadata


@dataclass(slots=True)
class GenerationRequest:
    """Normalised generation request.

    Both OpenAI and Anthropic payloads collapse into this shape before
    the engine sees them. Streaming callers iterate over the engine's
    output until they decide to stop (max_tokens, stop sequences,
    client disconnect).
    """

    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    stop: tuple[str, ...] = ()
    seed: int | None = None


class GenerationEngine:
    """One model, one executor, one tokenizer — owned for the process lifetime.

    Instantiate once during FastAPI startup. ``stream_chat`` and
    ``stream_complete`` are the only entry points HTTP handlers
    should call.
    """

    def __init__(
        self,
        model: Path,
        *,
        tokenizer: Path | None = None,
        cache_mb: int = 256,
        backend: str = "python",
        dequant_cache: bool = True,
    ) -> None:
        self._model_path = model
        self._cache_mb = cache_mb
        self._dequant_cache = dequant_cache
        # ``RuntimeConfig`` is the same one the CLI uses, with the
        # cache cap the user passed. ``probe=None`` matches the CLI
        # default — the memory manager falls back to its built-in
        # RSS probe.
        cfg = RuntimeConfig(memory=MemoryConfig(cache_bytes=cache_mb * 1024 * 1024, probe=None))
        # ``_resolve_model_paths`` is private to ``cli`` but its
        # behaviour (file vs directory, GGUF detection) is exactly
        # what we want here too. We import the function rather than
        # re-implementing so the two surfaces stay aligned.
        from argparse import ArgumentParser

        _stub = ArgumentParser()
        _stub.exit = lambda *_a, **_kw: None
        model_dir, gguf_path, fmt = _resolve_model_paths(_FakeArgs(model), _stub)
        if fmt == "gguf" and gguf_path is None:
            gguf_path = _pick_gguf_file(model_dir, _stub)
        self._format = fmt
        self._gguf_path = gguf_path
        self._tokenizer = self._build_tokenizer(model_dir, tokenizer, gguf_path, fmt)
        # The CLI's loader prints cache-bump messages. We mirror the
        # same behaviour here so a user running ``flatrun serve`` sees
        # the same "bumping cache from N MiB to M MiB" hint if their
        # model is bigger than the default.
        loaded = load_huggingface(model_dir, config=cfg)
        largest = max(
            (loaded.runtime.get_metadata(k.name).byte_size for k in loaded.runtime.list_tensors()),
            default=0,
        )
        if largest > 0 and cache_mb == 256:
            recommended_mb = max(
                256,
                ((largest * 4) + (128 * 1024 * 1024) - 1) // (128 * 1024 * 1024) * 128,
            )
            if recommended_mb > cache_mb:
                cache_mb = recommended_mb
                loaded.runtime.close()
                cfg = RuntimeConfig(
                    memory=MemoryConfig(cache_bytes=cache_mb * 1024 * 1024, probe=None)
                )
                loaded = load_huggingface(model_dir, config=cfg)
        self._loaded = loaded
        # Forwarder / scheduler / executor — same recipe as the CLI.
        if fmt == "gguf":
            raw_cfg = _build_config_from_gguf(gguf_path)
            qcfg = Qwen2Config.from_hf_config(raw_cfg)
            qcfg.quant_gguf = "Q8_0"
        else:
            if loaded.config is None or loaded.config.raw is None:
                raise RuntimeError("No config.json found next to model weights")
            qcfg = Qwen2Config.from_hf_config(loaded.config.raw)
            qcfg.quant_mlx_4bit = fmt == "mlx"
            qcfg.quant_gguf = None
        be = get_backend(backend)
        forwarder = make_qwen2_forwarder(
            qcfg,
            enable_dequant_cache=dequant_cache,
            backend=be,
        )
        scheduler = loaded.runtime.build_scheduler(
            loaded.manifest.layers,
            pre_layer_names=loaded.manifest.pre_layer,
            post_layer_names=loaded.manifest.post_layer,
        )
        self._executor = StreamingExecutor(scheduler, forwarder, kv_cache=KVCache(capacity=4096))
        # ``_lock`` serialises concurrent requests. Streaming LLMs
        # can't share an executor across interleaved requests because
        # each call resets the KV cache, so the safest policy for
        # v0 is "one request at a time". A request queue replaces
        # this once we have a true async forward pass.
        self._lock = threading.Lock()
        self._model_id = model_dir.name or (gguf_path.stem if gguf_path else "model")
        self._created = int(time.time())

    # ------------------------------------------------------------------
    # Properties exposed to the API layer
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def created(self) -> int:
        return self._created

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer.vocab)

    @property
    def max_context(self) -> int:
        cfg = self._loaded.config
        if cfg is not None and cfg.raw is not None:
            return int(cfg.raw.get("max_position_embeddings", 32768))
        return 32768

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def stream_chat(
        self,
        messages: list[dict],
        request: GenerationRequest,
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(kind, text)`` events for an OpenAI-style chat call.

        ``kind`` is either ``"reasoning"`` (inside a ``<think>`` block)
        or ``"text"`` (the user-visible reply). The Anthropic route
        splits these into separate content blocks; the OpenAI route
        emits ``reasoning`` content when the request asked for it.
        """
        prompt_text = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        yield from self._stream_with_thinking(prompt_text, request)

    def stream_complete(
        self,
        prompt: str,
        request: GenerationRequest,
    ) -> Iterator[str]:
        """Yield token strings for a legacy text-completion call.

        No chat template is applied — the prompt is sent through the
        BPE encoder directly, exactly as the OpenAI legacy endpoint
        expects.
        """
        for kind, text in self._stream_with_thinking(prompt, request):
            # Legacy completions ignore reasoning; emit everything as
            # text. The OpenAI route shapes them into the right JSON.
            yield text

    def _stream_with_thinking(
        self,
        prompt_text: str,
        request: GenerationRequest,
    ) -> Iterator[tuple[str, str]]:
        """Core loop: prefill once, then decode ``max_tokens`` steps.

        ``StreamingExecutor.step`` resets the KV cache per call and
        re-runs the full prefix; that matches the way the CLI is
        wired today. When a proper incremental decoder lands we
        swap this for ``step_incremental`` and the API layer doesn't
        need to change.

        The thinking extractor watches the decoded tokens for the
        ``<think>`` / ``</think>`` pair (Qwen3 reasoning format)
        and tags the events accordingly. Models that don't use
        reasoning tags emit everything as ``"text"``.
        """
        prompt_ids = self._tokenizer.encode(prompt_text)
        sampler = Sampler(
            temperature=request.temperature if request.temperature is not None else 0.05,
            top_k=request.top_k,
            top_p=request.top_p,
            min_p=0.0,
            repeat_penalty=1.0,
            seed=request.seed,
        )
        # The reason a lock wraps the whole loop and not just the
        # step is that we also mutate the executor's KV cache
        # between steps; another thread interleaving would corrupt
        # it. The lock is reentrant=False so a misbehaving handler
        # can't recursively call the engine.
        with self._lock:
            seen: list[int] = list(prompt_ids)
            generated: list[int] = []
            stop = {s for s in request.stop if s}
            # Pre-pend the generation tokens we've already produced to
            # the prompt for every re-run. ``step`` discards the KV
            # cache so we re-feed the full prefix; this is exactly
            # what the CLI does.
            next_id = -1
            state = "before"  # before | thinking | after
            buf = ""
            for _ in range(request.max_tokens):
                ids = prompt_ids + generated if next_id == -1 else prompt_ids + generated
                result = self._executor.step(tokens=ids)
                logits = result.last_hidden[-1]
                if request.temperature <= 0:
                    next_id = int(np.argmax(logits))
                else:
                    next_id = sampler.sample(logits, seen_ids=seen)
                generated.append(next_id)
                seen.append(next_id)
                text = self._tokenizer.decode([next_id])
                buf += text
                kind, payload = self._classify_token(state, buf)
                if payload is not None:
                    yield kind, payload
                    buf = ""
                # State advances only on tag transitions, see helper.
                state = self._advance_state(state, buf)
                # Stop sequences are checked AFTER the token is yielded
                # so a stop on the very last token still flushes it.
                for s in stop:
                    if s and self._ends_with_token(generated, s, self._tokenizer):
                        return
                # Common chat-model end-of-turn markers.
                if next_id in self._stop_token_ids():
                    return

    @staticmethod
    def _ends_with_token(generated: list[int], stop_str: str, tokenizer) -> bool:
        """True if decoding the tail of ``generated`` ends with ``stop_str``."""
        if not generated:
            return False
        # Cheap path: decode the last few tokens and string-match.
        tail_n = min(len(generated), 8)
        tail = tokenizer.decode(generated[-tail_n:])
        return tail.endswith(stop_str)

    # ------------------------------------------------------------------
    # Thinking-tag state machine
    # ------------------------------------------------------------------

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    @classmethod
    def _classify_token(cls, state: str, buf: str) -> tuple[str, str | None]:
        """Decide what to emit given the current state and buffered text.

        The buffer is the text accumulated since the last yield. The
        rule is: hold text in the buffer until we know whether it is
        part of a ``<think>`` / ``</think>`` tag, then yield the rest
        of the buffer tagged appropriately. ``cls`` makes this work
        when called as ``GenerationEngine._classify_token`` from a
        test on a stub that doesn't subclass ``GenerationEngine``.
        """
        if state == "before":
            if cls.OPEN_TAG in buf:
                idx = buf.index(cls.OPEN_TAG)
                pre = buf[:idx]
                rest = buf[idx + len(cls.OPEN_TAG) :]
                if cls.CLOSE_TAG in rest:
                    return cls._emit_thinking_then_text(rest)
                return "text", (pre or None)
            if cls.OPEN_TAG.startswith(buf) and len(buf) < len(cls.OPEN_TAG):
                return "text", None
            return "text", (buf or None)
        if state == "thinking":
            if cls.CLOSE_TAG in buf:
                idx = buf.index(cls.CLOSE_TAG)
                body = buf[:idx]
                tail = buf[idx + len(cls.CLOSE_TAG) :]
                yield_amt = ("reasoning", body) if body else ("text", None)
                if tail:
                    return "text", tail
                return yield_amt
            for i in range(len(cls.CLOSE_TAG), 0, -1):
                if buf.endswith(cls.CLOSE_TAG[:i]):
                    visible = buf[:-i]
                    return ("reasoning", visible) if visible else ("reasoning", None)
            return "reasoning", (buf or None)
        return "text", (buf or None)

    @classmethod
    def _emit_thinking_then_text(cls, rest: str) -> tuple[str, str | None]:
        if cls.CLOSE_TAG in rest:
            idx = rest.index(cls.CLOSE_TAG)
            tail = rest[idx + len(cls.CLOSE_TAG) :]
            return "text", (tail or None)
        return "reasoning", rest

    @classmethod
    def _advance_state(cls, state: str, buf: str) -> str:
        if state == "before":
            if cls.OPEN_TAG in buf:
                after = buf[buf.index(cls.OPEN_TAG) + len(cls.OPEN_TAG) :]
                if cls.CLOSE_TAG in after:
                    return "after"
                return "thinking"
            return state
        if state == "thinking":
            if cls.CLOSE_TAG in buf:
                return "after"
            return state
        return state

    def _stop_token_ids(self) -> set[int]:
        """Return token IDs the chat template treats as end-of-turn.

        Anything in the tokenizer's added-tokens table that looks
        like ``<|im_end|>`` or ``<|endoftext|>`` qualifies. Models
        with a custom stop token (e.g. Llama-3 ``<|eot_id|>``)
        benefit from the same scan.
        """
        stop_ids: set[int] = set()
        for tid, tok in getattr(self._tokenizer, "added_tokens", {}).items():
            if any(s in tok for s in ("im_end", "endoftext", "/s>", "end>", "eot_id")):
                stop_ids.add(int(tid))
        return stop_ids

    # ------------------------------------------------------------------
    # Tokenizer bootstrap
    # ------------------------------------------------------------------

    def _build_tokenizer(
        self,
        model_dir: Path,
        tokenizer: Path | None,
        gguf_path: Path | None,
        fmt: str,
    ):
        if tokenizer is not None:
            return auto_load(tokenizer)
        # GGUF dirs may not ship a sibling tokenizer — fall back to
        # building one from the GGUF metadata block.
        if fmt == "gguf" and gguf_path is not None and not any(
            (model_dir / f).is_file() for f in ("tokenizer.json", "vocab.json")
        ):
            return load_from_gguf_metadata(gguf_path)
        return auto_load(model_dir)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the runtime. Idempotent."""
        if self._loaded is not None:
            self._loaded.runtime.close()
            self._loaded = None  # type: ignore[assignment]


class _FakeArgs:
    """Tiny argparse stand-in so we can reuse CLI path-resolution helpers.

    ``_resolve_model_paths`` and ``_pick_gguf_file`` only look at
    ``args.model`` and call ``parser.exit`` on bad input. We swallow
    exit by passing a parser whose ``exit`` is a no-op so a bad path
    raises an honest exception here instead of ``SystemExit``.
    """

    def __init__(self, model: Path) -> None:
        self.model = model


__all__ = ["GenerationEngine", "GenerationRequest"]
