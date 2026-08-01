"""HTTP server exposing FlatRun via OpenAI- and Anthropic-compatible APIs.

This subpackage is intentionally dependency-free at import time:
``fastapi`` and ``uvicorn`` only load when :func:`build_app` is
called (so the base wheel stays light). The CLI ``flatrun serve``
pulls in the ``[serve]`` extra which installs FastAPI for you.

Endpoints implemented:

* ``GET  /v1/models``               (OpenAI compatible)
* ``POST /v1/chat/completions``     (OpenAI compatible, supports SSE)
* ``POST /v1/completions``          (OpenAI legacy text completions)
* ``POST /v1/messages``             (Anthropic compatible, supports SSE)
* ``GET  /healthz``                 (FlatRun's own liveness probe)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ServerConfig:
    """Configuration for :func:`build_app`.

    All paths may be a directory (HuggingFace / MLX / GGUF dir) or a
    single ``.gguf`` file path. The same detection rules the CLI uses
    apply here, so users can paste the path they already use in
    ``flatrun run`` straight into ``flatrun serve``.
    """

    model: Path
    tokenizer: Path | None = None
    cache_mb: int = 256
    backend: str = "python"  # "python" | "native"
    dequant_cache: bool = True
    host: str = "127.0.0.1"
    port: int = 8080


__all__ = ["ServerConfig", "build_app"]


def build_app(config: ServerConfig):
    """Construct the FastAPI application for ``config``.

    Lazy-imports FastAPI/uvicorn so importing :mod:`flatrun.serve`
    without the ``[serve]`` extra installed is safe.
    """
    from .app import build_app as _build_app

    return _build_app(config)
