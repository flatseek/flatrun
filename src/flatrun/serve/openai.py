"""OpenAI-compatible HTTP routes.

Wire format follows the OpenAI REST API reference (2024-Q4 snapshot):

* ``GET  /v1/models`` returns ``{"object": "list", "data": [...]}``.
* ``POST /v1/chat/completions`` accepts the standard ``messages`` array
  and emits either a single JSON response or an SSE stream
  (``text/event-stream`` with ``data: {...}`` frames terminated by
  ``data: [DONE]``).
* ``POST /v1/completions`` is the legacy text-completion endpoint:
  ``prompt`` is a string, no chat template.

The SSE encoding uses Pydantic models for the chunk shape so the
clients we ship with (``openai`` Python SDK >= 1.0) parse the stream
without extra configuration.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .engine import GenerationEngine, GenerationRequest
from .errors import BadRequestError, ContextLengthError, ModelNotLoadedError


router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None
    seed: int | None = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    max_tokens: int = 16
    stream: bool = False
    stop: list[str] | str | None = None
    seed: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stop_list(value) -> tuple[str, ...]:
    """Normalise the OpenAI ``stop`` field (str | list | None) to a tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _engine(request: Request) -> GenerationEngine:
    eng = getattr(request.app.state, "engine", None)
    if eng is None:
        raise ModelNotLoadedError()
    return eng


def _sse_format(data: dict) -> bytes:
    """Encode ``data`` as one SSE frame.

    OpenAI uses ``data: <json>\\n\\n`` per chunk with a literal
    ``data: [DONE]`` terminator. FastAPI's StreamingResponse writes
    bytes verbatim, so we build the wire format here.
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n".encode("utf-8")


def _sse_done() -> bytes:
    return b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    eng = _engine(request)
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": eng.model_id,
                    "object": "model",
                    "created": eng.created,
                    "owned_by": "flatrun",
                    "permission": [],
                }
            ],
        }
    )


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    eng = _engine(request)
    model_id = body.model or eng.model_id
    max_tokens = body.max_tokens if body.max_tokens is not None else 256
    if max_tokens <= 0:
        raise BadRequestError("max_tokens must be > 0")
    prompt_text = eng.tokenizer.apply_chat_template(
        [m.model_dump() for m in body.messages],
        add_generation_prompt=True,
    )
    prompt_ids = eng.tokenizer.encode(prompt_text)
    if len(prompt_ids) + max_tokens > eng.max_context(model_id):
        raise ContextLengthError(
            f"prompt ({len(prompt_ids)} tokens) + max_tokens ({max_tokens}) "
            f"exceeds model context ({eng.max_context(model_id)})"
        )
    req = GenerationRequest(
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=0,
        stop=_stop_list(body.stop),
        seed=body.seed,
    )
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_id = body.model or eng.model_id

    if not body.stream:
        text, reasoning = _collect_chat_text(eng, [m.model_dump() for m in body.messages], req, model_id)
        resp = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": 0,  # exact count unavailable without a token counter
                "total_tokens": len(prompt_ids),
            },
        }
        return JSONResponse(resp)

    # Streaming: emit one chunk per (reasoning, text) event from the
    # engine. Reasoning is sent as a single ``reasoning`` content
    # block at the end (the OpenAI spec doesn't define reasoning as
    # a streaming content part, so we collapse it). The visible text
    # is streamed token-by-token via ``content`` deltas.
    async def event_source():
        reasoning_buf: list[str] = []
        text_buf: list[str] = []
        # Send the role marker first; OpenAI clients use this to
        # open the assistant message.
        yield _sse_format(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
        for kind, payload in eng.stream_chat(
            [m.model_dump() for m in body.messages], req, model_id
        ):
            if kind == "reasoning":
                reasoning_buf.append(payload)
                # Surface the reasoning as it streams if the client
                # supports it. We use a custom ``reasoning_content``
                # delta — OpenAI's own spec doesn't define it yet but
                # the SDK ignores unknown keys, so this is forward-
                # compatible.
                yield _sse_format(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": payload},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            else:
                text_buf.append(payload)
                yield _sse_format(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": payload},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
        # Final chunk: ``finish_reason="stop"``.
        yield _sse_format(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        yield _sse_done()

    return StreamingResponse(event_source(), media_type="text/event-stream")


@router.post("/v1/completions")
async def completions(body: CompletionRequest, request: Request) -> Any:
    eng = _engine(request)
    if body.max_tokens <= 0:
        raise BadRequestError("max_tokens must be > 0")
    req = GenerationRequest(
        prompt=body.prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=0,
        stop=_stop_list(body.stop),
        seed=body.seed,
    )
    prompt_ids = eng.tokenizer.encode(body.prompt)
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_id = body.model or eng.model_id

    if not body.stream:
        text = "".join(eng.stream_complete(body.prompt, req, model_id))
        return JSONResponse(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "text": text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": 0,
                    "total_tokens": len(prompt_ids),
                },
            }
        )

    async def event_source():
        yield _sse_format(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": None}],
            }
        )
        for tok in eng.stream_complete(body.prompt, req, model_id):
            yield _sse_format(
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [{"text": tok, "index": 0, "logprobs": None, "finish_reason": None}],
                }
            )
        yield _sse_format(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}],
            }
        )
        yield _sse_done()

    return StreamingResponse(event_source(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Non-streaming collector
# ---------------------------------------------------------------------------


def _collect_chat_text(
    eng: GenerationEngine,
    messages: list[dict],
    req: GenerationRequest,
) -> tuple[str, str]:
    """Run the engine to completion and split reasoning from visible text.

    Used by the non-streaming ``/v1/chat/completions`` path. The
    OpenAI response only has a single ``content`` field, so we
    concatenate the reasoning with a leading ``"<think>...</think>"``
    block to preserve it (models that emit reasoning shouldn't lose
    it when the client uses the non-streaming endpoint). Reasoning-
    aware clients should use ``stream=True`` and the
    ``reasoning_content`` delta.
    """
    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    for kind, payload in eng.stream_chat(messages, req, model_id):
        if kind == "reasoning":
            reasoning_parts.append(payload)
        else:
            text_parts.append(payload)
    reasoning = "".join(reasoning_parts)
    text = "".join(text_parts)
    if reasoning:
        text = f"<think>{reasoning}</think>\n{text}"
    return text, reasoning


__all__ = ["router"]
