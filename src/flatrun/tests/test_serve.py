"""Tests for the ``flatrun serve`` HTTP server.

The tests stub out :class:`GenerationEngine` so they can exercise the
HTTP layer (route wiring, request validation, SSE framing, error
mapping) without loading a real model. Loading SmolLM2 here would
make the suite slow and network-dependent, which defeats the point
of unit tests.

The stub engine yields a fixed stream of events; the SSE tests
assert the wire format OpenAI / Anthropic clients expect, not the
exact tokens.
"""

from __future__ import annotations

from typing import Iterator

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient", reason="fastapi is not installed").TestClient

from flatrun.serve import ServerConfig, build_app  # noqa: E402
from flatrun.serve.engine import GenerationEngine, GenerationRequest  # noqa: E402
from flatrun.serve.errors import ServeError  # noqa: E402


# ---------------------------------------------------------------------------
# Stub engine
# ---------------------------------------------------------------------------


class StubTokenizer:
    """Minimal tokenizer stub. The router only asks it for vocab size."""

    vocab = {"hello": 0, "world": 1, "<|im_end|>": 2}
    inv_vocab = {0: "hello", 1: "world", 2: "<|im_end|>"}
    added_tokens = {2: "<|im_end|>"}
    chat_template = "{% for m in messages %}{{m['role']}}: {{m['content']}}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"

    def encode(self, text: str) -> list[int]:
        # Crude but deterministic — what the routes do is mostly count
        # tokens for the context-length check.
        return [0] * max(1, len(text.split()))

    def decode(self, ids) -> str:
        return self.inv_vocab.get(ids[0], "") if ids else ""

    def apply_chat_template(self, messages, *, add_generation_prompt: bool = True) -> str:
        # Tiny ChatML-ish renderer: enough for the route handlers
        # that just need a string to count tokens against.
        out: list[str] = []
        for m in messages:
            out.append(f"{m['role']}: {m['content']}")
        if add_generation_prompt:
            out.append("assistant:")
        return "\n".join(out)


class StubEngine:
    """Drop-in for :class:`GenerationEngine` with scripted streams."""

    def __init__(
        self,
        events: list[tuple[str, str]] | None = None,
        max_context: int = 4096,
        tokens_for_prompt: int = 4,
        raise_on_chat: Exception | None = None,
    ) -> None:
        self._events = events or [
            ("text", "Hello"),
            ("text", " world"),
            ("reasoning", "I should greet."),
            ("text", "!"),
        ]
        self._tokenizer = StubTokenizer()
        self._model_id = "stub-model"
        self._created = 1_700_000_000
        self._max_context = max_context
        self._tokens_for_prompt = tokens_for_prompt
        self._raise = raise_on_chat

    # Properties the routes read.
    model_id = property(lambda self: self._model_id)
    created = property(lambda self: self._created)
    tokenizer = property(lambda self: self._tokenizer)
    max_context = property(lambda self: self._max_context)
    vocab_size = property(lambda self: 3)

    def stream_chat(
        self,
        messages: list[dict],
        request: GenerationRequest,
    ) -> Iterator[tuple[str, str]]:
        if self._raise is not None:
            raise self._raise
        # Cap the stream at ``max_tokens`` so a stub doesn't blow up
        # the test budget.
        for kind, payload in self._events[: request.max_tokens]:
            yield kind, payload

    def stream_complete(
        self, prompt: str, request: GenerationRequest
    ) -> Iterator[str]:
        for kind, payload in self._events[: request.max_tokens]:
            # Legacy completions ignore reasoning.
            if kind == "text":
                yield payload

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_events() -> list[tuple[str, str]]:
    return [
        ("text", "Hello"),
        ("text", " world"),
        ("reasoning", "thinking aloud"),
        ("text", "!"),
    ]


@pytest.fixture()
def client(stub_events: list[tuple[str, str]]) -> TestClient:
    """A FastAPI TestClient whose engine is a deterministic stub."""
    from fastapi import FastAPI

    # We don't go through ``build_app`` here so the test doesn't try
    # to load a real model. Instead, we mount the routers on a fresh
    # app and inject the stub engine on ``app.state``.
    app = FastAPI()
    from flatrun.serve.openai import router as openai_router
    from flatrun.serve.anthropic import router as anthropic_router
    from flatrun.serve.app import serve_error_handler  # type: ignore

    app.include_router(openai_router)
    app.include_router(anthropic_router)
    app.add_exception_handler(ServeError, serve_error_handler)

    @app.get("/healthz")
    async def healthz(request):  # type: ignore[no-untyped-def]
        return {"status": "ready", "engine_loaded": True}

    app.state.engine = StubEngine(events=stub_events)
    return TestClient(app)


# ---------------------------------------------------------------------------
# OpenAI: /v1/models
# ---------------------------------------------------------------------------


def test_models_endpoint_lists_one(client: TestClient) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == "stub-model"
    assert body["data"][0]["object"] == "model"


# ---------------------------------------------------------------------------
# OpenAI: /v1/chat/completions (non-streaming)
# ---------------------------------------------------------------------------


def test_chat_completion_non_streaming(client: TestClient) -> None:
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "stub-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "stub-model"
    assert len(body["choices"]) == 1
    msg = body["choices"][0]["message"]
    # The stub has one reasoning event; the non-streaming path
    # preserves it as a <think> block.
    assert "Hello world" in msg["content"]
    assert "thinking aloud" in msg["content"]
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_rejects_zero_max_tokens(client: TestClient) -> None:
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "stub-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 0,
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"


def test_chat_completion_rejects_context_overflow(client: TestClient) -> None:
    # ``StubTokenizer.encode`` returns ~1 token per whitespace split
    # word. The stub's max_context is 4096; max_tokens=10000 blows
    # past it because the prompt itself contributes tokens.
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "stub-model",
            "messages": [{"role": "user", "content": "one two three"}],
            "max_tokens": 10000,
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["type"] == "context_length_exceeded"


# ---------------------------------------------------------------------------
# OpenAI: /v1/chat/completions (streaming)
# ---------------------------------------------------------------------------


def _sse_lines(body: str) -> list[dict]:
    """Parse an SSE body into the list of ``data:`` payloads."""
    import json as _json

    chunks: list[dict] = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            continue
        chunks.append(_json.loads(payload))
    return chunks


def test_chat_completion_streaming(client: TestClient) -> None:
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "stub-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    chunks = _sse_lines(r.text)
    assert chunks, "no SSE data frames"
    # First chunk must announce the assistant role.
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    # A reasoning chunk has ``reasoning_content``; a text chunk has
    # ``content``.
    saw_reasoning = any(
        "reasoning_content" in c["choices"][0]["delta"] for c in chunks
    )
    saw_text = any("content" in c["choices"][0]["delta"] for c in chunks)
    assert saw_reasoning and saw_text
    # Last chunk's finish_reason is "stop".
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    # Stream terminated with the OpenAI [DONE] sentinel.
    assert r.text.rstrip().endswith("data: [DONE]")


# ---------------------------------------------------------------------------
# OpenAI: /v1/completions
# ---------------------------------------------------------------------------


def test_completion_non_streaming(client: TestClient) -> None:
    r = client.post(
        "/v1/completions",
        json={
            "model": "stub-model",
            "prompt": "Once upon",
            "max_tokens": 4,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    # Legacy completions drop reasoning; "Hello world!" should be
    # the visible text.
    assert body["choices"][0]["text"] == "Hello world!"


def test_completion_streaming_uses_data_done(client: TestClient) -> None:
    r = client.post(
        "/v1/completions",
        json={
            "model": "stub-model",
            "prompt": "Once upon",
            "max_tokens": 4,
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    chunks = _sse_lines(r.text)
    assert chunks
    assert r.text.rstrip().endswith("data: [DONE]")


# ---------------------------------------------------------------------------
# Anthropic: /v1/messages
# ---------------------------------------------------------------------------


def test_anthropic_non_streaming(client: TestClient) -> None:
    r = client.post(
        "/v1/messages",
        json={
            "model": "stub-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "system": "Be brief.",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    # Anthropic uses content blocks; reasoning goes in a thinking
    # block, text in a text block.
    blocks = body["content"]
    kinds = [b["type"] for b in blocks]
    assert "thinking" in kinds
    assert "text" in kinds
    text_block = next(b for b in blocks if b["type"] == "text")
    assert "Hello world!" in text_block["text"]
    assert body["stop_reason"] == "end_turn"


def test_anthropic_streaming_uses_typed_events(client: TestClient) -> None:
    r = client.post(
        "/v1/messages",
        json={
            "model": "stub-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events: list[tuple[str, dict]] = []
    import json as _json

    for line in r.text.splitlines():
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            payload = line[len("data:") :].strip()
            if payload:
                events.append((current_event, _json.loads(payload)))
    names = [name for name, _ in events]
    # Required Anthropic SSE event sequence.
    assert names[0] == "message_start"
    assert "content_block_start" in names
    assert "content_block_delta" in names
    assert names[-1] == "message_stop"
    # We have a thinking block AND a text block.
    starts = [e for n, e in events if n == "content_block_start"]
    block_types = {s["content_block"]["type"] for s in starts}
    assert block_types == {"thinking", "text"}


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def test_anthropic_zero_max_tokens_returns_anthropic_shape(client: TestClient) -> None:
    r = client.post(
        "/v1/messages",
        json={
            "model": "stub-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 0,
        },
    )
    assert r.status_code == 400
    body = r.json()
    # ``/v1/messages`` routes through the Anthropic error shape.
    assert body.get("type") == "error"
    assert body["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Engine-level unit tests (no HTTP)
# ---------------------------------------------------------------------------


def test_engine_classify_state_machine_simple() -> None:
    """The thinking-tag state machine routes text vs reasoning correctly."""
    # ``before`` + open tag present -> emits the pre-text as text.
    kind, payload = GenerationEngine._classify_token("before", "hello<think>hi")
    assert kind == "text"
    assert payload == "hello"
    # In "thinking" with close-tag forming at tail -> hold.
    kind, payload = GenerationEngine._classify_token("thinking", "deep thought")
    assert kind == "reasoning"
    assert payload == "deep thought"
    # In "thinking" with full close tag -> emit body then transition.
    kind, payload = GenerationEngine._classify_token("thinking", "deep thought</think>now")
    assert payload == "now"
    # ``before`` with prefix of the open tag -> hold (None payload).
    kind, payload = GenerationEngine._classify_token("before", "<thi")
    assert payload is None


def test_engine_advance_state_transitions() -> None:
    assert GenerationEngine._advance_state("before", "<<think>x</think>y") == "after"
    assert GenerationEngine._advance_state("before", "<think") == "before"
    assert GenerationEngine._advance_state("thinking", "deep thought") == "thinking"
    assert GenerationEngine._advance_state("thinking", "deep</think>ok") == "after"


def test_server_config_roundtrip() -> None:
    """``ServerConfig`` is a dataclass — exercise the constructor shape."""
    from pathlib import Path

    cfg = ServerConfig(model=Path("/tmp/model"), port=9000)
    assert cfg.port == 9000
    assert cfg.cache_mb == 256
    assert cfg.backend == "python"
