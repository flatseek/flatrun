"""Anthropic-compatible HTTP routes.

Anthropic's Messages API has its own SSE shape that differs from
OpenAI's. Each event is a typed JSON object (``message_start``,
``content_block_start``, ``content_block_delta``,
``content_block_stop``, ``message_delta``, ``message_stop``) and the
stream is terminated by ``message_stop`` (NOT a ``[DONE]`` sentinel).

The reasoning content goes into a separate ``content_block`` of
type ``"thinking"`` so the official SDK's content-block router
treats it the same as text.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .engine import GenerationEngine, GenerationRequest
from .errors import BadRequestError, ContextLengthError, ModelNotLoadedError


router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------


class AnthropicMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class MessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine(request: Request) -> GenerationEngine:
    eng = getattr(request.app.state, "engine", None)
    if eng is None:
        raise ModelNotLoadedError()
    return eng


def _sse_event(event: str, data: dict) -> bytes:
    """Anthropic SSE: ``event: <name>\\ndata: <json>\\n\\n``."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _to_chatml(messages: list[dict], system: str | None) -> list[dict]:
    """Translate Anthropic ``{role, content}`` into the OpenAI shape the engine expects.

    Anthropic keeps the system message outside the ``messages`` array;
    we prepend it so the chat template renders once.
    """
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        out.append({"role": m["role"], "content": m["content"]})
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/v1/messages")
async def messages(body: MessagesRequest, request: Request) -> Any:
    eng = _engine(request)
    if body.max_tokens <= 0:
        raise BadRequestError("max_tokens must be > 0")
    chatml = _to_chatml([m.model_dump() for m in body.messages], body.system)
    prompt_text = eng.tokenizer.apply_chat_template(chatml, add_generation_prompt=True)
    prompt_ids = eng.tokenizer.encode(prompt_text)
    if len(prompt_ids) + body.max_tokens > eng.max_context(model_id):
        raise ContextLengthError(
            f"prompt ({len(prompt_ids)} tokens) + max_tokens ({body.max_tokens}) "
            f"exceeds model context ({eng.max_context(model_id)})"
        )
    req = GenerationRequest(
        prompt=prompt_text,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p if body.top_p is not None else 1.0,
        top_k=body.top_k if body.top_k is not None else 0,
        stop=tuple(body.stop_sequences or ()),
    )
    msg_id = f"msg_{uuid.uuid4().hex}"
    model_id = body.model or eng.model_id

    if not body.stream:
        text, reasoning = _collect_anthropic_text(eng, chatml, req, model_id)
        resp: dict[str, Any] = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model_id,
            "content": [],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": len(prompt_ids), "output_tokens": 0},
        }
        if reasoning:
            resp["content"].append({"type": "thinking", "thinking": reasoning})
        if text:
            resp["content"].append({"type": "text", "text": text})
        return JSONResponse(resp)

    # Streaming: emit two content blocks (reasoning, then text) using
    # Anthropic's typed event sequence. Block indices are 0 (thinking)
    # and 1 (text). A small detail: Anthropic's SDK expects the
    # ``input_json`` deltas to be valid JSON; reasoning is plain text
    # so we use ``text_delta`` semantics on a thinking block.
    async def event_source():
        # ``message_start`` first; carries the message id and empty
        # usage. The ``content`` block list is empty here and is
        # filled in by the subsequent ``content_block_start``.
        yield _sse_event(
            "message_start",
            {
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model_id,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": len(prompt_ids), "output_tokens": 0},
                }
            },
        )
        # ``ping`` is a keep-alive some clients expect. Emit one early
        # so an empty-generation request doesn't hang on the first
        # byte.
        yield _sse_event("ping", {})
        # Walk the engine stream. We open the thinking block the first
        # time we see a reasoning token and the text block the first
        # time we see a visible token; both stay open until the model
        # is done (or until a ``stop_sequence`` matches).
        thinking_open = False
        text_open = False
        for kind, payload in eng.stream_chat(chatml, req, model_id):
            if kind == "reasoning" and not thinking_open:
                yield _sse_event(
                    "content_block_start",
                    {"index": 0, "content_block": {"type": "thinking", "thinking": ""}},
                )
                thinking_open = True
            if kind == "text" and not text_open:
                yield _sse_event(
                    "content_block_start",
                    {"index": 1, "content_block": {"type": "text", "text": ""}},
                )
                text_open = True
            block_index = 0 if kind == "reasoning" else 1
            yield _sse_event(
                "content_block_delta",
                {
                    "index": block_index,
                    "delta": (
                        {"type": "thinking_delta", "thinking": payload}
                        if kind == "reasoning"
                        else {"type": "text_delta", "text": payload}
                    ),
                },
            )
        # Close any open blocks before the final message_delta.
        if thinking_open:
            yield _sse_event("content_block_stop", {"index": 0})
        if text_open:
            yield _sse_event("content_block_stop", {"index": 1})
        # ``message_delta`` carries the stop reason. Anthropic's
        # SDK uses ``stop_reason="end_turn"`` to know generation
        # completed without a stop-sequence match.
        yield _sse_event(
            "message_delta",
            {"delta": {"stop_reason": "end_turn", "stop_sequence": None}},
        )
        yield _sse_event("message_stop", {})

    return StreamingResponse(event_source(), media_type="text/event-stream")


def _collect_anthropic_text(
    eng: GenerationEngine,
    chatml: list[dict],
    req: GenerationRequest,
    model_id: str | None = None,
) -> tuple[str, str]:
    """Run the engine to completion and split reasoning vs text."""
    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    for kind, payload in eng.stream_chat(chatml, req, model_id):
        if kind == "reasoning":
            reasoning_parts.append(payload)
        else:
            text_parts.append(payload)
    return "".join(text_parts), "".join(reasoning_parts)


__all__ = ["router"]
