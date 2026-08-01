# `flatrun serve` — HTTP API server

`flatrun serve` exposes a loaded FlatRun model over HTTP with two
shapes of API on the same port:

* **OpenAI-compatible** — the same `/v1/models`,
  `/v1/chat/completions`, and `/v1/completions` endpoints the
  OpenAI Python SDK (>= 1.0) and most local LLM front-ends talk to.
* **Anthropic-compatible** — `/v1/messages` with the typed SSE event
  format (`message_start`, `content_block_start`,
  `content_block_delta`, `content_block_stop`, `message_delta`,
  `message_stop`).

The server is a thin wrapper around the same
[`GenerationEngine`](src/flatrun/serve/engine.py) the CLI uses — it
loads the model once on startup, holds a single `StreamingExecutor`
under a lock, and runs the request loop on the HTTP thread.

## Install

The server needs FastAPI + uvicorn. They ship as the `[serve]`
extra so the base wheel stays light:

```bash
pip install 'flatrun[serve]'
# or, for the native backend too:
pip install 'flatrun[serve,native]'
```

## Run

```bash
flatrun serve --model /path/to/qwen2.5-0.5b-instruct-gguf --port 8080
```

Useful flags (shared with `flatrun run` / `flatrun chat`):

| Flag | Default | Notes |
| --- | --- | --- |
| `--model` | (required) | Path to a directory or a single `.gguf` file |
| `--tokenizer` | model dir | Override tokenizer location |
| `--cache-mb` | 256 | Memory cache cap |
| `--backend` | `python` | `python` (NumPy) or `native` (C++/NEON) |
| `--dequant-cache` | `on` | `off` for true streaming on memory-constrained hosts |
| `--host` | `127.0.0.1` | Loopback only by default; use `0.0.0.0` to expose on LAN |
| `--port` | `8080` | TCP port |

The server prints the bound address on startup. The first request
pays the model-load cost during the FastAPI startup handler so the
HTTP listener is ready before the load completes.

## OpenAI-compatible API

`/v1/models`

```bash
curl -s http://127.0.0.1:8080/v1/models | jq
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen2.5-Coder-0.5B-Instruct-Q4_K_M",
      "object": "model",
      "created": 1700000000,
      "owned_by": "flatrun",
      "permission": []
    }
  ]
}
```

`/v1/chat/completions` (streaming, OpenAI SDK compatible):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
stream = client.chat.completions.create(
    model="Qwen2.5-Coder-0.5B-Instruct-Q4_K_M",
    messages=[{"role": "user", "content": "def fib(n):"}],
    max_tokens=128,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

`/v1/completions` (legacy text completions):

```bash
curl -s http://127.0.0.1:8080/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "Qwen2.5-Coder-0.5B-Instruct-Q4_K_M",
      "prompt": "def hello():",
      "max_tokens": 64,
      "stream": true
    }'
```

Both endpoints emit SSE (`text/event-stream`) with `data: {...}`
frames when `stream: true`, terminated by a literal `data: [DONE]`
sentinel that the OpenAI SDK recognises.

## Anthropic-compatible API

`/v1/messages`:

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8080", api_key="not-needed")
with client.messages.stream(
    model="Qwen2.5-Coder-0.5B-Instruct-Q4_K_M",
    max_tokens=128,
    system="You are a careful refactorer.",
    messages=[{"role": "user", "content": "Rewrite this loop as a comprehension."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

The Anthropic SDK expects the typed SSE event sequence. FlatRun
emits all six (`message_start`, `ping`, `content_block_start`,
`content_block_delta`, `content_block_stop`, `message_delta`,
`message_stop`). Reasoning content from Qwen3-style models is
routed to a separate `thinking` content block so the SDK's
content-block router picks it up automatically.

## Reasoning content

Both APIs surface the model's chain-of-thought separately from the
visible reply:

* **OpenAI streaming** — reasoning is emitted as a `delta` field
  named `reasoning_content`. The OpenAI SDK ignores unknown fields,
  so this is forward-compatible with whatever the spec settles on.
  In the non-streaming response, reasoning is preserved verbatim
  inside the assistant `content` as a `<think>...</think>` block.
* **Anthropic streaming** — reasoning is opened as a `content_block`
  of type `"thinking"`, with `delta.type == "thinking_delta"`. The
  Anthropic SDK already routes these to its thinking-block helpers.

Models that don't use `<think>...</think>` reasoning tags (most
Qwen2.5, Llama-3, Mistral) emit everything as plain text.

## Errors

Both surfaces share the same JSON error shape (status code, type
discriminator, message). Examples:

```json
// OpenAI
{"error": {"message": "max_tokens must be > 0",
           "type": "invalid_request_error",
           "param": null, "code": 400}}
```

```json
// Anthropic
{"type": "error",
 "error": {"type": "invalid_request_error",
           "message": "max_tokens must be > 0"}}
```

Status codes follow the OpenAI spec: `400` for malformed input,
`413` for prompt that exceeds the model's context window, `503`
during startup before the model finishes loading.

## Concurrency

The first version of the server handles one request at a time.
Streaming LLM inference can't share a single `StreamingExecutor`
across interleaved requests because each call resets the KV cache
— the lock in `GenerationEngine._lock` serialises access. A request
queue with bounded concurrency is on the roadmap; the engine
interface won't change when it lands.

## Python API

If you want to embed the server in your own application instead of
launching `flatrun serve` as a subprocess:

```python
import uvicorn
from flatrun.serve import ServerConfig, build_app

config = ServerConfig(model="/path/to/model", port=9000)
app = build_app(config)
uvicorn.run(app, host=config.host, port=config.port)
```

`build_app` constructs the FastAPI app and wires the model-load
lifecycle (load on startup, close on shutdown). The engine is
stored on `app.state.engine` so you can also call into it from
custom routes.
