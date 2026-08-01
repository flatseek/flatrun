"""Exception → HTTP status mapping for the serve routes.

Both OpenAI and Anthropic clients expect a JSON ``{"error": ...}``
body when something goes wrong, with a status code that matches the
caller's retry semantics. Centralising the mapping here keeps the
route handlers short and lets us swap response shapes per API
without re-implementing the error logic.
"""

from __future__ import annotations


class ServeError(Exception):
    """Base class for all serve errors.

    ``status_code`` is what FastAPI will return; ``kind`` is the
    discriminator clients use (``"invalid_request_error"`` for OpenAI,
    ``"invalid_request_error"`` for Anthropic). ``message`` is what
    the human reads.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        kind: str = "invalid_request_error",
        extra: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.kind = kind
        self.message = message
        self.extra = extra or {}

    def to_openai(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": None,
                "code": self.status_code,
            }
        }

    def to_anthropic(self) -> dict:
        return {"type": "error", "error": {"type": self.kind, "message": self.message}}


class BadRequestError(ServeError):
    """Missing / malformed field on the request body."""

    def __init__(self, message: str, **kw) -> None:
        super().__init__(message, status_code=400, **kw)


class ContextLengthError(ServeError):
    """Prompt + max_tokens would exceed the model's context window."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400, kind="context_length_exceeded")


class ModelNotLoadedError(ServeError):
    """Engine was never initialised. Should be unreachable in practice."""

    def __init__(self) -> None:
        super().__init__("model not loaded", status_code=503, kind="server_error")
