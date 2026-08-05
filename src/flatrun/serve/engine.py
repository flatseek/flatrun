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


class _ModelInstance:
    """Loaded model with its executor and tokenizer."""
    
    def __init__(
        self,
        model_id: str,
        loaded,
        tokenizer,
        executor: StreamingExecutor,
    ) -> None:
        self.model_id = model_id
        self.loaded = loaded
        self.tokenizer = tokenizer
        self.executor = executor


class GenerationEngine:
    """Multiple models, multiple executors — owned for the process lifetime.

    Instantiate once during FastAPI startup. Supports multiple GGUF files
    in a directory. ``stream_chat`` and ``stream_complete`` select the
    model based on request.
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
        self._tokenizer_path = tokenizer
        self._backend = backend
        self._models: dict[str, _ModelInstance] = {}
        self._default_model_id: str | None = None
        self._created = int(time.time())
        
        # Load models from path
        self._load_models()

    def _load_models(self) -> None:
        """Load all GGUF files from the model path."""
        from argparse import ArgumentParser

        _stub = ArgumentParser()
        _stub.exit = lambda *_a, **_kw: None
        
        model_path = Path(self._model_path)
        
        if model_path.is_file() and model_path.suffix == ".gguf":
            # Single GGUF file
            gguf_path = model_path
            model_dir = model_path.parent
            model_id = model_path.stem
            self._load_single_model(model_id, model_dir, gguf_path, _stub)
            self._default_model_id = model_id
        elif model_path.is_dir():
            # Directory - load all GGUF files
            gguf_files = sorted(model_path.glob("*.gguf"))
            if not gguf_files:
                raise FileNotFoundError(f"No .gguf files in {model_path}")
            
            for gguf_path in gguf_files:
                model_id = gguf_path.stem
                self._load_single_model(model_id, model_path, gguf_path, _stub)
            
            self._default_model_id = gguf_files[0].stem
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")

    def _load_single_model(
        self,
        model_id: str,
        model_dir: Path,
        gguf_path: Path,
        stub,
    ) -> None:
        """Load a single GGUF model."""
        cfg = RuntimeConfig(memory=MemoryConfig(cache_bytes=self._cache_mb * 1024 * 1024, probe=None))
        
        fmt = "gguf"
        tokenizer = self._build_tokenizer(model_dir, self._tokenizer_path, gguf_path, fmt)
        
        loaded = load_huggingface(gguf_path, config=cfg)
        largest = max(
            (loaded.runtime.get_metadata(k.name).byte_size for k in loaded.runtime.list_tensors()),
            default=0,
        )
        if largest > 0 and self._cache_mb == 256:
            recommended_mb = max(
                256,
                ((largest * 4) + (128 * 1024 * 1024) - 1) // (128 * 1024 * 1024) * 128,
            )
            if recommended_mb > self._cache_mb:
                loaded.runtime.close()
                cfg = RuntimeConfig(
                    memory=MemoryConfig(cache_bytes=recommended_mb * 1024 * 1024, probe=None)
                )
                loaded = load_huggingface(gguf_path, config=cfg)
        
        raw_cfg = _build_config_from_gguf(gguf_path)
        qcfg = Qwen2Config.from_hf_config(raw_cfg)
        qcfg.quant_gguf = "Q8_0"
        
        be = get_backend(self._backend)
        forwarder = make_qwen2_forwarder(
            qcfg,
            enable_dequant_cache=self._dequant_cache,
            backend=be,
        )
        scheduler = loaded.runtime.build_scheduler(
            loaded.manifest.layers,
            pre_layer_names=loaded.manifest.pre_layer,
            post_layer_names=loaded.manifest.post_layer,
        )
        executor = StreamingExecutor(scheduler, forwarder, kv_cache=KVCache(capacity=4096))
        
        self._models[model_id] = _ModelInstance(model_id, loaded, tokenizer, executor)
        self._locks = {mid: threading.Lock() for mid in self._models}

    def get_model(self, model_id: str | None) -> _ModelInstance:
        """Get model instance by ID, or default."""
        if model_id and model_id in self._models:
            return self._models[model_id]
        if self._default_model_id:
            return self._models[self._default_model_id]
        raise ValueError("No model loaded")

    # ------------------------------------------------------------------
    # Properties exposed to the API layer
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        """Default model ID."""
        return self._default_model_id or "model"

    @property
    def created(self) -> int:
        return self._created

    @property
    def tokenizer(self):
        """Default tokenizer."""
        default = self.get_model(None)
        return default.tokenizer

    def vocab_size(self, model_id: str | None = None) -> int:
        return len(self.get_model(model_id).tokenizer.vocab)

    def max_context(self, model_id: str | None = None) -> int:
        cfg = self.get_model(model_id).loaded.config
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
        model_id: str | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(kind, text)`` events for an OpenAI-style chat call.

        ``kind`` is either ``"reasoning"`` (inside a ``<think>`` block)
        or ``"text"`` (the user-visible reply). The Anthropic route
        splits these into separate content blocks; the OpenAI route
        emits ``reasoning`` content when the request asked for it.
        """
        inst = self.get_model(model_id)
        prompt_text = inst.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        yield from self._stream_with_thinking(prompt_text, request, inst)

    def stream_complete(
        self,
        prompt: str,
        request: GenerationRequest,
        model_id: str | None = None,
    ) -> Iterator[str]:
        """Yield token strings for a legacy text-completion call.

        No chat template is applied — the prompt is sent through the
        BPE encoder directly, exactly as the OpenAI legacy endpoint
        expects.
        """
        for kind, text in self._stream_with_thinking(prompt, request, self.get_model(model_id)):
            # Legacy completions ignore reasoning; emit everything as
            # text. The OpenAI route shapes them into the right JSON.
            yield text

    def _stream_with_thinking(
        self,
        prompt_text: str,
        request: GenerationRequest,
        inst: _ModelInstance,
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
        prompt_ids = inst.tokenizer.encode(prompt_text)
        sampler = Sampler(
            temperature=request.temperature or 1.0,
            top_k=request.top_k,
            top_p=request.top_p,
            min_p=0.0,
            repeat_penalty=1.0,
            seed=request.seed,
        )
        lock = self._locks[inst.model_id]
        # The reason a lock wraps the whole loop and not just the
        # step is that we also mutate the executor's KV cache
        # between steps; another thread interleaving would corrupt
        # it. The lock is reentrant=False so a misbehaving handler
        # can't recursively call the engine.
        with lock:
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
                result = inst.executor.step(tokens=ids)
                logits = result.last_hidden[-1]
                if request.temperature <= 0:
                    next_id = int(np.argmax(logits))
                else:
                    next_id = sampler.sample(logits, seen_ids=seen)
                generated.append(next_id)
                seen.append(next_id)
                text = inst.tokenizer.decode([next_id])
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
                    if s and self._ends_with_token(generated, s, inst.tokenizer):
                        return
                # Common chat-model end-of-turn markers.
                if next_id in self._stop_token_ids(inst.tokenizer):
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

    def _stop_token_ids(self, tokenizer) -> set[int]:
        """Return token IDs the chat template treats as end-of-turn.

        Anything in the tokenizer's added-tokens table that looks
        like ``<|im_end|>`` or ``<|endoftext|>`` qualifies. Models
        with a custom stop token (e.g. Llama-3 ``<|eot_id|>``)
        benefit from the same scan.
        """
        stop_ids: set[int] = set()
        for tid, tok in getattr(tokenizer, "added_tokens", {}).items():
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
