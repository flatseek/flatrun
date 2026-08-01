"""FastAPI application factory for ``flatrun serve``.

Wiring is deliberately small:

* :class:`GenerationEngine` is constructed once during the startup
  event and stored on ``app.state.engine``.
* Routers from :mod:`flatrun.serve.openai` and
  :mod:`flatrun.serve.anthropic` are mounted under their respective
  prefixes.
* A single :class:`ServeError` exception handler normalises error
  responses across both API shapes. The handler picks the right
  error JSON shape (``to_openai`` or ``to_anthropic``) by sniffing
  the request path.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import ServerConfig
from .engine import GenerationEngine
from .errors import ServeError


async def serve_error_handler(request: Request, exc: ServeError) -> JSONResponse:
    """Module-level exception handler so tests can register it on a bare app.

    Mounted on the FastAPI app produced by :func:`build_app`; the
    test fixture also mounts it on the bare test app so the same
    error-shape logic applies whether the app was built via the
    public factory or assembled by hand.
    """
    if request.url.path.startswith("/v1/messages"):
        body = exc.to_anthropic()
    else:
        body = exc.to_openai()
    return JSONResponse(body, status_code=exc.status_code)


def build_app(config: ServerConfig) -> FastAPI:
    """Construct the FastAPI application for ``config``."""
    app = FastAPI(
        title="flatrun serve",
        description=(
            "OpenAI- and Anthropic-compatible HTTP interface to a local "
            "FlatRun streaming model. The same model directory works for "
            "both surfaces."
        ),
        version="0.1.1",
    )

    @app.on_event("startup")
    def _startup() -> None:
        # Lazy-construct on startup so import-time doesn't require a
        # valid model path. The CLI passes a real path; tests pass
        # a stub and override ``app.state.engine`` before the first
        # request.
        app.state.engine = GenerationEngine(
            config.model,
            tokenizer=config.tokenizer,
            cache_mb=config.cache_mb,
            backend=config.backend,
            dequant_cache=config.dequant_cache,
        )
        app.state.server_config = config

    @app.on_event("shutdown")
    def _shutdown() -> None:
        eng = getattr(app.state, "engine", None)
        if eng is not None:
            eng.close()

    @app.get("/healthz")
    async def healthz(request: Request) -> JSONResponse:
        eng = getattr(request.app.state, "engine", None)
        ready = eng is not None
        return JSONResponse(
            {"status": "ready" if ready else "loading", "engine_loaded": ready},
            status_code=200 if ready else 503,
        )

    # Routers. OpenAI uses ``/v1`` prefix (so paths are
    # ``/v1/models``, ``/v1/chat/completions``). Anthropic's spec
    # uses the same ``/v1`` prefix; we mount the Anthropic router
    # separately so its event names don't collide with the OpenAI
    # SSE format.
    from .openai import router as openai_router
    from .anthropic import router as anthropic_router

    app.include_router(openai_router)
    app.include_router(anthropic_router)
    app.add_exception_handler(ServeError, serve_error_handler)

    @app.middleware("http")
    async def _access_log(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        dt_ms = (time.perf_counter() - t0) * 1000
        # One line per request, stderr — same shape as uvicorn's
        # default access log so users can grep either.
        import sys

        sys.stderr.write(
            f"[flatrun.serve] {request.method} {request.url.path} "
            f"-> {response.status_code} {dt_ms:.1f}ms\n"
        )
        sys.stderr.flush()
        return response

    return app


__all__ = ["build_app"]
