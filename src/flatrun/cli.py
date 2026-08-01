"""Unified FlatRun CLI.

Run a single prompt (or a chat turn) through any FlatRun-supported model:

* GGUF directory (``*.gguf``)
* SafeTensors directory (``model.safetensors`` + ``config.json``)
* MLX-4bit directory (weight + scales + biases triplets)

The CLI auto-detects the format, derives a Qwen2 config from GGUF
metadata when no ``config.json`` is present, and applies the model's
chat template (or the Qwen2 ChatML default) when ``--messages`` is
provided.

Examples::

    # GGUF
    PYTHONPATH=src python examples/flatrun_chat.py \\
        --model /Users/judotens/.lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF \\
        --prompt "def hello():"

    # SafeTensors
    PYTHONPATH=src python examples/flatrun_chat.py \\
        --model /Users/judotens/Works/.../qwen2.5-0.5b \\
        --prompt "Once upon a time"

    # MLX-4bit
    PYTHONPATH=src python examples/flatrun_chat.py \\
        --model /Users/judotens/.lmstudio/models/lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit \\
        --prompt "def hello():" \\
        --max-new 4

    # Multi-turn chat
    PYTHONPATH=src python examples/flatrun_chat.py \\
        --model /path/to/qwen \\
        --messages-json '[{"role":"system","content":"You are concise."},{"role":"user","content":"Hi"}]'
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

from flatrun import (
    KVCache,
    RuntimeConfig,
    StreamingExecutor,
    load_huggingface,
)
from flatrun.model.qwen2 import Qwen2Config, make_qwen2_forwarder
from flatrun.model.sampling import Sampler
from flatrun.runtime.memory import MemoryConfig
from flatrun.tokenizer import auto_load
from flatrun.utils.errors import ConfigurationError


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Spinner + thinking extraction
# ---------------------------------------------------------------------------
#
# Inference on a Qwen2.5-class model is slow enough that a static prompt
# makes the user think the CLI has hung. The :class:`Spinner` runs a
# lightweight animation on stderr (a single carriage-returned line that
# redraws every 100 ms) while the forwarder is busy, and disappears
# cleanly when generation finishes - either with a newline that drops the
# animated line, or with the final assistant text that overwrites it.
#
# Modern reasoning models (Qwen3, DeepSeek-R1 distilled, ...) emit a
# ``<think>...</think>`` block before the user-visible reply. We split
# the two so the chat REPL can show the chain-of-thought in dim colour
# during generation and the clean answer afterwards.


_SPINNER_FRAMES = "-\\|/"
_SPINNER_INTERVAL = 0.1  # seconds

# ANSI colour codes used by the chat REPL. Centralised so the chat
# output is consistent and easy to tweak without grepping the
# call sites. ``\033[0m`` resets the active style.
_C_USER = "\033[36m"     # cyan   - "You:" label
_C_USER_END = "\033[0m"
_C_ASSISTANT = "\033[32m"  # green  - "Assistant:" label
_C_ASSISTANT_END = "\033[0m"
_C_DIM = "\033[2m"          # dim
_C_GREY = "\033[97m"        # bright-white - "grey" thinking body (lighter than 90)
_C_ITALIC = "\033[3m"       # italic  - thinking body
_C_YELLOW = "\033[33m"     # yellow  - live cursor
_C_END = "\033[0m"

# Braille-pattern frames for the live cursor that pulses at the
# LEFT of the line being streamed. The yellow frame replaces the
# thinking marker, so the user sees a spinning yellow dot
# followed by a dim 'Thinking:' label while the model is still
# reasoning. The frame advances each tick; ``\r`` rewinds the
# line so the rest of the streamed content is not overwritten.
_LIVE_CURSOR_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_LIVE_CURSOR_INTERVAL = 0.12  # seconds


class Spinner:
    """Single-line, stderr-only animation that lives for the duration
    of a generation step.

    Usage::

        with Spinner("Thinking"):
            ...do slow work...

    The animated line stays out of the way of stdout and is overwritten
    by whatever the caller prints next.
    """

    def __init__(self, label: str = "Thinking", stream=None) -> None:
        self._label = label
        self._stream = stream or sys.stderr
        self._thread = None
        self._stop = False

    def __enter__(self) -> "Spinner":
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=_SPINNER_INTERVAL * 3)
        # Clear the spinner line so the caller can print over it cleanly.
        try:
            self._stream.write("\r" + " " * (len(self._label) + 4) + "\r")
            self._stream.flush()
        except Exception:
            pass

    def _run(self) -> None:
        i = 0
        while not self._stop:
            frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
            try:
                self._stream.write(f"\r{frame} {self._label}...")
                self._stream.flush()
            except Exception:
                return
            i += 1
            time.sleep(_SPINNER_INTERVAL)


class LiveCursor:
    """Pulsing braille cursor at the end of a streaming token line.

    Used during ``--max-new`` generation: tokens stream on ``stdout``
    in place, and this class writes a single dim braille frame to
    ``stderr`` via ``\\r`` every ``_LIVE_CURSOR_INTERVAL`` seconds.
    The cursor sits at the bottom-right of the user's terminal
    (where their eye is) and pulses while the model is still
    thinking, so the response never looks frozen. When ``__exit__``
    runs the cursor is cleared with a single overwrite and the
    finished transcript starts on a clean line.
    """

    def __init__(self, stream=None) -> None:
        self._stream = stream or sys.stderr
        self._thread = None
        self._stop = False

    def __enter__(self) -> "LiveCursor":
        self._stop = False
        # Use ANSI absolute positioning to put the cursor on a fixed
        # row at the bottom of the screen (``\033[999;1H`` moves to
        # row 999, column 1 - any reasonable terminal has fewer
        # than 999 lines so this is "off the bottom of the visible
        # content"). This decouples the cursor from stdout's
        # streaming cursor, which is the bug that produced
        # ``⠏ssistant:`` when both streams happened to share the
        # same TTY line.
        try:
            self._stream.write("\033[999;1H")
            self._stream.flush()
        except Exception:
            pass
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=_LIVE_CURSOR_INTERVAL * 3)
        # Clear the cursor row (``\033[K`` = clear from cursor to end
        # of line) and rewind to the bottom of the screen so the
        # next stderr write - the ``\n`` in ``cmd_chat`` after the
        # streamed content - lands on a fresh line below everything.
        try:
            self._stream.write("\033[999;1H\033[K")
            self._stream.flush()
        except Exception:
            pass

    def _run(self) -> None:
        i = 0
        while not self._stop:
            frame = _LIVE_CURSOR_FRAMES[i % len(_LIVE_CURSOR_FRAMES)]
            try:
                # Yellow cursor at column 0 of the reserved bottom
                # row. The leading ``\033[999;1H`` re-asserts the
                # absolute position each tick so a stray ``\n`` from
                # the streamed-content path can't push the cursor
                # away from the reserved row.
                self._stream.write(
                    f"\033[999;1H{_C_YELLOW}{frame}{_C_END}"
                )
                self._stream.flush()
            except Exception:
                return
            i += 1
            time.sleep(_LIVE_CURSOR_INTERVAL)


class LiveThinkingDisplay:
    """Animated cursor + scrolling thinking content during streaming.

    Renders a single line on stderr (the line lives right under the
    streamed answer):

        [yellow cursor] [dim]Thinking:[end] [grey italic]content[end]

    ``feed_token`` is the streaming callback: it inspects each
    decoded token, detects the ``<think ...>`` open and ``</think>``
    close markers, and accumulates the body. The body grows
    monotonically until it exceeds ``MAX_WORDS``; the OLDEST words
    are only dropped once the buffer has cleared that threshold, so
    short chains of thought stay visible in full.

    Two design constraints drove the rendering strategy:

    * **Don't flicker the line on every token.** Earlier iterations
      rewrote the whole line with ``\\r\\x1b[K`` per token and the
      user read that as "the thinking is always being cleared". The
      new renderer paints the line once on activation, then **appends**
      new chunks in place; the cursor tick only repaints the trailing
      frame cell.
    * **Don't show a fake Thinking line on non-reasoning models.**
      Before the open tag is seen we paint a single placeholder
      line - dim "Thinking:" + a literal "..." body - that pulses
      without ever being rewritten. If the model never opens a
      think block, ``__exit__`` simply blanks that one line.

    The cursor cell and the content cell are physically separate, so
    a tick that only swaps the braille frame never disturbs the
    accumulated buffer.
    """

    OPEN_TAG = "<" + "think" + ">"
    CLOSE_TAG = "<" + "/" + "think" + ">"
    MAX_WORDS = 15  # 10-20 words before the window scrolls
    PLACEHOLDER_BODY = "..."  # shown on the thinking line for non-reasoning models

    def __init__(self, stream=None) -> None:
        self._stream = stream or sys.stderr
        self._thread = None
        self._stop = False
        self._raw_buffer = ""
        self._content = ""
        self._state = "before"  # before | thinking | after
        self._lock = threading.Lock()
        # ``_active`` distinguishes "we've painted the Thinking line"
        # from "we haven't touched stderr yet". It is flipped on the
        # first paint and never cleared mid-stream - the line stays
        # live until ``__exit__`` blanks it.
        self._active = False
        # ``_rendered_len`` is the length of ``self._content`` (in
        # characters) at the moment we last painted it. The per-token
        # appender uses this to know which slice of ``_content`` is
        # new and must be written after the cursor was last drawn.
        self._rendered_len = 0
        # ``_rendered_frame`` is the braille frame currently on screen
        # at the cursor cell. The animation tick advances this in
        # place without touching the content.
        self._rendered_frame = _LIVE_CURSOR_FRAMES[0]
        # ``_placeholder_on_screen`` is True while the live line shows
        # the literal ``...`` placeholder (non-reasoning models, or
        # the very first moments of a reasoning model before any
        # body has arrived). The first time real content arrives, we
        # must do one full rewrite to drop the placeholder - the
        # appender otherwise would leave ``...real content`` on
        # screen.
        self._placeholder_on_screen = False

    def __enter__(self) -> "LiveThinkingDisplay":
        # Always start the animation thread: for non-reasoning models
        # the placeholder line must still pulse. The thread itself
        # gates on ``_active`` so the very first tick is the one that
        # paints the placeholder.
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=_LIVE_CURSOR_INTERVAL * 3)
        # If we ever painted the Thinking line, blank it so the
        # next assistant turn starts on a clean slate. The
        # transcript copy of the thinking block is printed by
        # ``cmd_chat`` after the stream finishes, so we don't lose
        # any reasoning by blanking the live view here.
        if self._active:
            with self._lock:
                try:
                    self._stream.write("\r\x1b[K")
                    self._stream.flush()
                except Exception:
                    pass

    def _activate(self) -> None:
        """Paint the Thinking line the first time the line goes live.

        This is called from the animation thread's first tick
        (``_active`` flips from False to True and we paint in the
        same critical section). It may also be called from
        ``feed_token`` once the model emits ``<think ...>``, in
        which case the animation thread has already painted the
        placeholder - we just upgrade the body to the empty
        (reasoning) state, which the next per-token append will
        fill out.

        NOTE: the caller MUST hold ``self._lock``.
        """
        if self._active:
            return
        self._active = True
        self._rendered_frame = _LIVE_CURSOR_FRAMES[0]
        self._placeholder_on_screen = True
        try:
            self._stream.write(
                f"\n{_C_DIM}Thinking:{_C_END} "
                f"{_C_GREY}{_C_ITALIC}{self.PLACEHOLDER_BODY}{_C_END}"
                f"{_C_YELLOW}{self._rendered_frame}{_C_END}"
            )
            self._stream.flush()
            self._rendered_len = len(self.PLACEHOLDER_BODY)
        except Exception:
            pass

    def feed_token(self, text: str) -> None:
        """Handle one decoded token from the streaming executor.

        State machine:

        * ``before`` - the open tag has not been seen yet. Text is
          buffered rather than streamed straight to stdout, because
          a token like ``<`` could be the start of ``<think ...>``.
          Once the buffer is longer than the open tag and the tag
          still isn't present, the buffered text is safe to flush.
        * ``thinking`` - the open tag was seen, the close tag is not
          yet. The body is the text between the open tag and the
          end of the buffer, minus any prefix of the close tag that
          might be forming at the tail. That body is appended to the
          live line in place; once the word count exceeds
          ``MAX_WORDS`` we do one full rewrite with the truncated
          tail.
        * ``after`` - the close tag was seen. Later tokens are the
          assistant's reply and go to stdout.
        """
        import re as _re
        self._raw_buffer += text
        with self._lock:
            if self._state == "before":
                if self.OPEN_TAG in self._raw_buffer:
                    open_idx = self._raw_buffer.index(self.OPEN_TAG)
                    prefix = self._raw_buffer[:open_idx]
                    if prefix:
                        sys.stdout.write(prefix)
                        sys.stdout.flush()
                    # The model is reasoning - flip the live line on
                    # if the animation thread hasn't already. We may
                    # already have the placeholder on screen; that
                    # is fine, the first append below will start
                    # appending the real reasoning right after it.
                    self._activate()
                    self._flush_to_thinking_or_after()
                elif (
                    len(self._raw_buffer) < len(self.OPEN_TAG)
                    and self.OPEN_TAG.startswith(self._raw_buffer)
                ):
                    # Buffer is a prefix of the open tag - hold it
                    # silently until the next token tells us whether
                    # the model is actually reasoning. This is the
                    # ONE place we delay the stream: the only case
                    # where the buffered text could grow into
                    # ``<think ...>``.
                    pass
                else:
                    # Buffer is either longer than the open tag, or
                    # it can no longer grow into the open tag - in
                    # both cases it's safe to flush to stdout
                    # immediately. Non-reasoning models never
                    # trigger the prefix branch above, so their
                    # tokens flow straight to the terminal as
                    # ``sys.stdout.write`` per token. The live
                    # Thinking line is already on screen with its
                    # placeholder body, and will be blanked on
                    # ``__exit__``.
                    sys.stdout.write(self._raw_buffer)
                    sys.stdout.flush()
                    self._raw_buffer = ""
            elif self._state == "thinking":
                if self.CLOSE_TAG in self._raw_buffer:
                    self._extract_thinking()
                    self._state = "after"
                    close_idx = self._raw_buffer.index(self.CLOSE_TAG) + len(self.CLOSE_TAG)
                    tail = self._raw_buffer[close_idx:]
                    if tail:
                        sys.stdout.write(tail)
                        sys.stdout.flush()
                    self._raw_buffer = ""
                else:
                    open_idx = self._raw_buffer.index(self.OPEN_TAG) + len(self.OPEN_TAG)
                    body = self._raw_buffer[open_idx:]
                    # Strip any prefix of the close tag that is
                    # forming at the tail so the live display never
                    # shows ``</`` or ``</think`` as part of the
                    # thinking content.
                    for i in range(len(self.CLOSE_TAG), 0, -1):
                        if body.endswith(self.CLOSE_TAG[:i]):
                            body = body[:-i]
                            break
                    self._content = _re.sub(r"\s+", " ", body).strip()
                    # Truncate by word count: keep the most recent
                    # ``MAX_WORDS`` words once the buffer has
                    # cleared the threshold. Splitting on whitespace
                    # is fine here because ``_re.sub(r"\s+", " ", ...)``
                    # above already collapsed runs of whitespace into
                    # a single space. Short chains of thought never
                    # hit the truncation branch and stay visible in
                    # full.
                    words = self._content.split(" ")
                    if len(words) > self.MAX_WORDS:
                        new_content = " ".join(words[-self.MAX_WORDS :])
                        # Truncation requires a full rewrite of the
                        # live line - the only place we still use
                        # ``\r\x1b[K``. This is rare (the buffer
                        # has to clear MAX_WORDS first) so it does
                        # not look like the line is being cleared
                        # per token.
                        self._content = new_content
                        self._full_render()
                    else:
                        # The buffer is still inside the word budget.
                        # Only append the slice that hasn't been
                        # painted yet - this is what stops the line
                        # from flickering per token.
                        self._append_render()
            else:  # after
                sys.stdout.write(text)
                sys.stdout.flush()

    def _flush_to_thinking_or_after(self) -> None:
        """After the open tag is in the buffer, decide between
        ``thinking`` and ``after`` based on whether the close tag is
        already present too.
        """
        after_open = self._raw_buffer[
            self._raw_buffer.index(self.OPEN_TAG) + len(self.OPEN_TAG) :
        ]
        if self.CLOSE_TAG in after_open:
            self._extract_thinking()
            self._state = "after"
            close_idx = self._raw_buffer.index(self.CLOSE_TAG) + len(self.CLOSE_TAG)
            tail = self._raw_buffer[close_idx:]
            if tail:
                sys.stdout.write(tail)
                sys.stdout.flush()
            self._raw_buffer = ""
        else:
            self._state = "thinking"

    def _extract_thinking(self) -> None:
        import re as _re
        start = self._raw_buffer.index(self.OPEN_TAG) + len(self.OPEN_TAG)
        end = self._raw_buffer.index(self.CLOSE_TAG)
        body = self._raw_buffer[start:end]
        body = _re.sub(r"\s+", " ", body).strip()
        words = body.split(" ")
        if len(words) > self.MAX_WORDS:
            body = " ".join(words[-self.MAX_WORDS :])
        self._content = body
        # Whole block arrived in one shot - one full render is fine.
        self._full_render()

    def _full_render(self) -> None:
        """Repaint the entire thinking line. Used when truncation
        happens or when the placeholder line first transitions to
        real content.
        """
        self._placeholder_on_screen = False
        try:
            self._stream.write(
                f"\r\x1b[K{_C_DIM}Thinking:{_C_END} "
                f"{_C_GREY}{_C_ITALIC}{self._content}{_C_END}"
                f"{_C_YELLOW}{self._rendered_frame}{_C_END}"
            )
            self._stream.flush()
            self._rendered_len = len(self._content)
        except Exception:
            pass

    def _append_render(self) -> None:
        """Append only the freshly accumulated characters onto the
        live line, then advance the cursor cell.

        The streaming cursor is already at the end of the line after
        the previous paint - either just after the body or just after
        the cursor frame, depending on whether the animation thread
        has ticked since the last paint. We rewind one cell (so the
        new write lands on top of the old cursor frame, not after
        it) and emit ``<new chunk><updated cursor frame>``.

        If the placeholder is still on screen and the body now has
        real content, fall back to a full rewrite - the appender
        would otherwise leave ``...real content`` visible.
        """
        if self._placeholder_on_screen and self._content:
            self._full_render()
            return
        new_chars = self._content[self._rendered_len :]
        if not new_chars:
            # The buffer might have been collapsed by whitespace
            # normalisation without gaining any visible characters.
            # Still need to advance the cursor cell so it doesn't
            # freeze on an old frame.
            self._advance_cursor()
            return
        try:
            # ``\b`` rewinds one column so the new write overwrites
            # the trailing cursor frame rather than appending past
            # it. The grey italic run is closed (back to default
            # attributes) at the end of the body chunk so the yellow
            # cursor frame keeps its colour.
            self._stream.write(
                f"\b{_C_GREY}{_C_ITALIC}{new_chars}{_C_END}"
                f"{_C_YELLOW}{self._rendered_frame}{_C_END}"
            )
            self._stream.flush()
            self._rendered_len += len(new_chars)
        except Exception:
            pass

    def _advance_cursor(self) -> None:
        """Rewrite just the trailing cursor frame, leaving the
        content untouched. Cheap enough to run on every animation
        tick (the buffer is empty so we only emit two bytes).
        """
        try:
            self._stream.write(
                f"\b{_C_YELLOW}{self._rendered_frame}{_C_END}"
            )
            self._stream.flush()
        except Exception:
            pass

    def _run(self) -> None:
        i = 0
        # First tick: flip the line live if it isn't already. This
        # is what makes the placeholder appear at the start of
        # generation - we don't have to know in advance whether the
        # model is reasoning or not.
        first_tick = True
        while not self._stop:
            frame = _LIVE_CURSOR_FRAMES[i % len(_LIVE_CURSOR_FRAMES)]
            with self._lock:
                if first_tick:
                    first_tick = False
                    # If ``feed_token`` already activated the line
                    # before the thread got its first tick, do
                    # nothing extra here - the placeholder is
                    # already on screen.
                    if not self._active:
                        self._activate()
                        # Skip this tick's frame advance; the
                        # cursor was just painted.
                        i += 1
                        time.sleep(_LIVE_CURSOR_INTERVAL)
                        continue
                if frame != self._rendered_frame:
                    self._rendered_frame = frame
                    self._advance_cursor()
            i += 1
            time.sleep(_LIVE_CURSOR_INTERVAL)


# Patterns we recognise as "this is a thinking block". The first match
# wins; everything outside it is the user-visible reply. Order matters
# because the Qwen3 format wraps the whole trace in
# ``<|im_start|>think\\n...<|im_start|>assistant`` and we want the
# human-readable answer to start at the second marker.
_THINK_PATTERNS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<|thinking|>", "<|/thinking|>"),
)


def _split_thinking(text: str) -> tuple[str | None, str]:
    """Pull a chain-of-thought block out of ``text``.

    Returns ``(thinking, answer)``. If no block is found,
    ``thinking`` is ``None`` and the whole input is the answer.

    The ``<think>`` and ``</think>`` markers themselves are stripped
    from the returned thinking string; only the body remains. The
    body is run through :func:`_flatten_thinking` so multi-line chain
    of thought renders as a single line in the REPL.
    """
    for open_tag, close_tag in _THINK_PATTERNS:
        start = text.find(open_tag)
        if start == -1:
            continue
        end = text.find(close_tag, start + len(open_tag))
        if end == -1:
            # Unterminated block - treat everything after ``open_tag``
            # as thinking and the rest as answer (best effort).
            thinking = _flatten_thinking(text[start + len(open_tag) :])
            answer = text[:start].rstrip()
            return (thinking or None), answer
        thinking = _flatten_thinking(text[start + len(open_tag) : end])
        # Everything before the block plus everything after it.
        answer = text[:start] + text[end + len(close_tag) :]
        return (thinking or None), answer.lstrip()
    return None, text


def _flatten_thinking(s: str) -> str:
    """Return ``s`` with runs of whitespace collapsed to a single space.

    Used by :func:`_split_thinking` to render a multi-line chain of
    thought as a single ``[dim][italic][grey]`` line in the REPL
    rather than as a tall block that scrolls the transcript
    off-screen. Trailing / leading whitespace is removed. If the
    input is empty the result is empty too.
    """
    import re as _re
    return _re.sub(r"\s+", " ", s).strip() or ""


def _detect_format(model_dir: Path, gguf_path: Path | None = None) -> str:
    """Return one of ``"gguf"``, ``"safetensors"``, ``"mlx"``.

    MLX-4bit stores its ``weight``/``scales``/``biases`` triplets either
    as a single multi-tensor safetensors file or as separate ``*.weight``
    / ``*.scales`` / ``*.biases`` files. Either form counts as MLX.

    ``gguf_path`` overrides the directory scan and forces a GGUF
    detection - used when the user passes a file path on the CLI.
    """
    if gguf_path is not None or any(model_dir.glob("*.gguf")):
        return "gguf"
    if (model_dir / "model.safetensors").is_file() or any(model_dir.glob("*.safetensors")):
        if _looks_like_mlx_4bit(model_dir):
            return "mlx"
        return "safetensors"
    raise FileNotFoundError(f"No .gguf or .safetensors file in {model_dir}")


def _looks_like_mlx_4bit(model_dir: Path) -> bool:
    """Return ``True`` if any weight has the MLX ``weight/scales/biases`` triplet.

    For a single ``model.safetensors`` we open it briefly and look for
    any tensor whose name ends with ``.scales`` (the MLX format always
    pairs a ``.weight`` with a ``.scales`` and ``.biases``).
    """
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        return False
    # Fast path: if any sibling file is ``*.scales`` or ``*.biases``,
    # the directory is definitely MLX-style.
    if any(model_dir.glob("*.scales")) or any(model_dir.glob("*.biases")):
        return True
    # Otherwise open the first safetensors file and inspect the header.
    try:
        from flatrun.backend.safetensor import open_safetensors
        backend = open_safetensors(safetensors_files[0])
        backend.open()
        try:
            for k in backend.list_tensors():
                if k.name.endswith(".scales") or k.name.endswith(".biases"):
                    return True
        finally:
            backend.close()
    except Exception:
        return False
    return False


# GGUF architectures that ggml rotates with LLAMA_ROPE_TYPE_NORM, i.e.
# over pairs of *consecutive* head dimensions. ``convert_hf_to_gguf.py``
# un-permutes Q/K for exactly these, so the on-disk weights need the
# interleaved rotation rather than HuggingFace's rotate_half. Every
# other architecture (qwen2, phi3, gemma, stablelm, ...) is NEOX and
# keeps the HF layout. Mirrors ``llama_rope_type()`` in llama.cpp.
_GGUF_ROPE_NORM_ARCHES = frozenset(
    {
        "llama",
        "baichuan",
        "starcoder",
        "plamo",
        "orion",
        "internlm2",
        "minicpm",
        "xverse",
        "command-r",
        "olmo",
        "arctic",
        "deepseek2",
        "chatglm",
    }
)


def _build_config_from_gguf(gguf_path: Path) -> dict:
    """Derive an HF-shaped config dict from a GGUF's metadata block.

    GGUF namespaces its hyperparameters under the architecture name
    (``qwen2.block_count``, ``llama.block_count``, ...), so the prefix
    is read from ``general.architecture`` rather than assumed.
    """
    from flatrun.backend.gguf import GGUFBackend
    backend = GGUFBackend(gguf_path)
    backend.open()
    try:
        meta = backend.gguf_metadata
        arch = str(meta.get("general.architecture") or "").strip()
        if not arch:
            raise ConfigurationError(
                f"{gguf_path.name}: GGUF metadata has no 'general.architecture'"
            )

        def need(suffix: str):
            key = f"{arch}.{suffix}"
            if key not in meta:
                raise ConfigurationError(
                    f"{gguf_path.name}: GGUF metadata is missing {key!r}"
                )
            return meta[key]

        n_heads = int(need("attention.head_count"))
        hidden = int(need("embedding_length"))
        cfg = {
            "vocab_size": len(meta.get("tokenizer.ggml.tokens", []) or []),
            "hidden_size": hidden,
            "intermediate_size": int(need("feed_forward_length")),
            "num_hidden_layers": int(need("block_count")),
            "num_attention_heads": n_heads,
            "num_key_value_heads": int(
                meta.get(f"{arch}.attention.head_count_kv", n_heads)
            ),
            "rope_theta": float(meta.get(f"{arch}.rope.freq_base", 10000.0)),
            "rms_norm_eps": float(need("attention.layer_norm_rms_epsilon")),
            "max_position_embeddings": int(
                meta.get(f"{arch}.context_length", 32768)
            ),
            # GGUF only emits a separate output projection when the LM
            # head is untied; absence means it reuses the embedding.
            "tie_word_embeddings": not any(
                k.name in ("lm_head.weight", "output.weight")
                for k in backend.list_tensors()
            ),
        }
        head_dim = meta.get(f"{arch}.attention.key_length")
        if head_dim is not None:
            cfg["head_dim"] = int(head_dim)
        elif hidden % n_heads:
            raise ConfigurationError(
                f"{gguf_path.name}: hidden_size={hidden} is not divisible by "
                f"head_count={n_heads} and no key_length was provided"
            )
        cfg["rope_interleaved"] = arch in _GGUF_ROPE_NORM_ARCHES
        return cfg
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _assemble_prompt(
    tokenizer,
    *,
    messages_json: str | None,
    system: str | None,
    prompt: str | None,
) -> str:
    """Return the text to encode.

    Priority:

    1. ``--messages-json`` (list of {role, content} dicts) - rendered
       through the chat template.
    2. ``--system`` + ``--prompt`` (single user turn) - rendered.
    3. ``--prompt`` alone - returned as-is.
    """
    if messages_json:
        messages = json.loads(messages_json)
        if not isinstance(messages, list):
            raise ValueError("--messages-json must be a JSON list")
        return tokenizer.apply_chat_template(messages)
    if system or prompt is not None:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt or ""})
        return tokenizer.apply_chat_template(msgs)
    return prompt or ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import sys
    if argv is None:
        argv = sys.argv[1:]
    parser, _ = _build_argparser()
    # Backwards-compat: pre-existing scripts invoke
    # ``flatrun --model ... --prompt ...`` without a subcommand.
    # Detect that and prepend ``run`` so the legacy CLI works.
    if len(argv) == 0:
        argv = ["run"]
    elif argv[0] not in ("run", "chat", "-h", "--help"):
        argv = ["run", *argv]
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler == "chat":
        return cmd_chat(args)
    return cmd_run(args)


def _build_argparser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Build the top-level parser and the shared parent parser.

    The shared parent carries every flag that's identical between
    ``run`` (one-shot) and ``chat`` (interactive REPL) - model path,
    tokenizer, runtime cache, sampling knobs, chat-template controls.
    """
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a model directory (GGUF, SafeTensors, or MLX-4bit).",
    )
    shared.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Tokenizer directory. Defaults to --model. Used for GGUF dirs that ship no tokenizer.",
    )
    shared.add_argument(
        "--system",
        type=str,
        default=None,
        help="Optional system message - prepended in chat templates.",
    )
    shared.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Skip the chat template; treat prompts as raw text.",
    )
    shared.add_argument(
        "--cache-mb",
        type=int,
        default=256,
        help="Memory cache cap in MiB (lower = more streaming).",
    )
    shared.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Tokens to generate after the prompt. run: default 1; chat: default 128.",
    )
    shared.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top tokens to show for the next-token prediction.",
    )
    shared.add_argument(
        "--quant",
        type=str,
        default=None,
        help="Override GGUF quant name. Default: detect from filename.",
    )
    shared.add_argument(
        "--temperature",
        type=float,
        default=0.11,
        help="Sampling temperature. Lower = more deterministic. 0.0 not allowed.",
    )
    shared.add_argument(
        "--sample-top-k",
        type=int,
        default=20,
        help="Sample-time top-k filter (0 disables).",
    )
    shared.add_argument(
        "--sample-top-p",
        type=float,
        default=0.59,
        help="Sample-time nucleus filter. 1.0 disables.",
    )
    shared.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        help="Sample-time min-p filter. 0.0 disables.",
    )
    shared.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty applied to seen token logits. 1.0 disables.",
    )
    shared.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy argmax (skip all sampling).",
    )
    shared.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for sampling. Default: time-seeded.",
    )
    shared.add_argument(
        "--profile",
        action="store_true",
        help="Print per-step timing breakdown for the first N generation steps.",
    )
    shared.add_argument(
        "--profile-detailed",
        action="store_true",
        help="Print per-layer microsecond breakdown of every forward-"
             "pass operation (RMSNorm, QKV projection, QK matmul, "
             "softmax, AV matmul, MLP, ...). Use this to find the "
             "real performance bottleneck in the forward pass.",
    )
    shared.add_argument(
        "--profile-save",
        type=str,
        default=None,
        metavar="PATH",
        help="Write the detailed profiler result to PATH as JSON. "
             "Implies --profile-detailed.",
    )
    shared.add_argument(
        "--debug",
        action="store_true",
        help="Print per-layer hidden-state norms and position-collapse "
             "metrics to stderr. Useful for cross-checking a model's "
             "forwarder against a reference (LM Studio, llama.cpp). "
             "With --debug, the CLI also emits a per-token debug "
             "table per layer and a Layer Analysis Summary at the "
             "end of inference.",
    )
    shared.add_argument(
        "--debug-include-special",
        action="store_true",
        help="When set with --debug, special tokens (BOS, EOS, chat "
             "markers, ...) are shown in the per-token debug table. "
             "By default they are filtered out so the table focuses "
             "on content tokens.",
    )
    shared.add_argument(
        "--debug-save-analysis",
        type=str,
        default=None,
        metavar="PATH",
        help="Write the Layer Analysis Summary (per-layer scores, "
             "most active / most stable ranks, suggested early-exit "
             "and suggested layer subset) to PATH as JSON. Useful "
             "for cross-prompt research on selective layer execution. "
             "Implies --debug.",
    )
    shared.add_argument(
        "--debug-max-token-rows",
        type=int,
        default=16,
        metavar="N",
        help="Maximum number of tokens to show in the per-token "
             "debug table per layer (default 16). Set higher to "
             "inspect every token in long prompts.",
    )
    shared.add_argument(
        "--dequant-cache",
        choices=["on", "off"],
        default="off",
        help="Keep dequantized weight tensors in Python heap across "
             "layers. ``off`` (default) is true streaming: every layer "
             "is dequantized fresh and the result is released as soon "
             "as the layer finishes, so the Python heap stays nearly "
             "constant. ``on`` trades RAM for speed by caching every "
             "dequantized weight until the process exits. Set ``on`` "
             "only for short-running experiments on small models.",
    )
    shared.add_argument(
        "--memory-trace",
        action="store_true",
        help="Print OS RSS, Python heap, mmap resident, dequant cache, "
             "and KV cache size after every decoder block. Use with "
             "--dequant-cache off to verify streaming behaviour.",
    )
    shared.add_argument(
        "--compare-layer",
        type=str,
        default=None,
        metavar="REF_JSON",
        help="Compare per-layer hidden-state statistics against a "
             "reference JSON (output of --debug, captured from LM "
             "Studio, llama.cpp, or HF transformers). Dumps FlatRun's "
             "per-layer stats to <path>.flatrun.json and reports the "
             "first layer where any of mean / std / L2 / cos_to_prev / "
             "row_cos diverges by more than the configured tolerance. "
             "Use this to localise where the forwarder drifts from "
             "a reference runtime without manually diffing the trace.",
    )
    shared.add_argument(
        "--compare-tol",
        type=float,
        default=0.05,
        metavar="TOL",
        help="Tolerance for --compare-layer. A layer is flagged when "
             "any relative divergence exceeds this fraction. Default 0.05 "
             "(5 percent).",
    )
    shared.add_argument(
        "--max-layers",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N decoder layers. Equivalent to "
             "running a truncated-depth copy of the model: the first N "
             "layers run in order, embeddings are loaded with the first "
             "selected layer, and the final norm + LM head fire on the "
             "Nth selected layer. Mutually exclusive with --layers.",
    )
    shared.add_argument(
        "--layers",
        type=str,
        default=None,
        metavar="LIST",
        help="Run a custom subset of decoder layers in the given order. "
             "Accepts a comma-separated list of 0-indexed indices and "
             "inclusive ranges, e.g. '1,3,4,6,7,8' or '0-6,19-24,34-39'. "
             "Whitespace is ignored. Duplicates are removed while "
             "preserving order. The hidden state passes through the "
             "selected layers in order; unlisted layers are skipped. The "
             "first selected layer embeds tokens and the last selected "
             "layer applies the final norm + LM head. Mutually exclusive "
             "with --max-layers.",
    )

    parser = argparse.ArgumentParser(
        prog="flatrun",
        description=(
            "FlatRun - streaming inference runtime. "
            "Use 'flatrun run' for one-shot prompts or 'flatrun chat' "
            "for an interactive REPL. Without a subcommand, the "
            "legacy one-shot ``run`` mode is used."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    # ``required=False`` keeps ``parser.parse_args`` happy when no
    # subcommand was given. ``main()`` rewrites the argv to inject
    # ``run`` before parsing, so the absent-subcommand case never
    # actually reaches the parser.

    # ``run`` - the original one-shot command. Defaults to being
    # invoked when no subcommand is given.
    run_parser = subparsers.add_parser(
        "run",
        parents=[shared],
        add_help=True,
        help="Run a single prompt and print the continuation.",
    )
    run_parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Plain text prompt. Ignored if --messages-json is set.",
    )
    run_parser.add_argument(
        "--messages-json",
        type=str,
        default=None,
        help="JSON list of {role, content} dicts. Rendered via the chat template.",
    )
    run_parser.set_defaults(_handler="run")

    # ``chat`` - interactive REPL.
    chat_parser = subparsers.add_parser(
        "chat",
        parents=[shared],
        add_help=True,
        help="Interactive REPL: type prompts, read assistant replies until EOF.",
    )
    chat_parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not append previous turns to the prompt. Each reply is a one-shot call.",
    )
    chat_parser.set_defaults(_handler="chat")

    return parser, shared


def _resolve_model_paths(args, parser: argparse.ArgumentParser) -> tuple[Path, Path | None, str]:
    """Resolve the model path, optional GGUF path, and detected format."""
    model_path: Path = args.model
    if not model_path.exists():
        parser.exit(1, f"Model path not found: {model_path}\n")
    if model_path.is_file():
        if model_path.suffix.lower() != ".gguf":
            parser.exit(
                1,
                "FlatRun only accepts a single .gguf file as a file path; "
                f"got {model_path.name}. For other formats point at the "
                "directory containing the model.safetensors files.\n",
            )
        return model_path.parent, model_path, "gguf"
    return model_path, None, _detect_format(model_path)


def _pick_gguf_file(model_dir: Path, parser: argparse.ArgumentParser) -> Path:
    """Pick the LLM GGUF in a directory, skipping multimodal helpers.

    LM Studio stores vision-language models with two GGUF files
    side by side: the base LLM (e.g. ``Bonsai-27B-Q1_0.gguf``)
    and the multimodal projection adapter
    (e.g. ``Bonsai-27B-mmproj-BF16.gguf``). flatrun is text-only,
    so loading the mmproj file would waste memory and produce
    garbage. The picker silently skips files whose stem contains
    ``mmproj`` / ``mm-proj`` / ``vision`` / ``clip`` / ``projection``;
    if only ``mmproj`` files exist, fall through and let the user
    see the original 'multiple GGUF files' error so they can
    correct the path manually.
    """
    skip_patterns = (
        "mmproj",
        "mm-proj",
        "mm_proj",
        "vision",
        "clip",
        "projection",
        "imgproj",
    )

    def is_llm(p: Path) -> bool:
        stem = p.stem.lower()
        return not any(pat in stem for pat in skip_patterns)

    candidates = sorted(model_dir.glob("*.gguf"))
    if not candidates:
        parser.exit(1, f"No .gguf file in {model_dir}\n")
    llm_candidates = [p for p in candidates if is_llm(p)]
    if len(llm_candidates) == 1:
        return llm_candidates[0]
    if len(candidates) > 1 and not llm_candidates:
        # Only mmproj-shaped files exist - tell the user to point
        # at the base model explicitly.
        parser.exit(
            1,
            f"All GGUF files in {model_dir} look like multimodal "
            f"projection helpers ({[p.name for p in candidates]}); "
            f"flatrun is text-only. Point --model at the base model "
            f"file directly.\n",
        )
    if len(llm_candidates) > 1:
        parser.exit(
            1,
            f"Multiple LLM GGUF files in {model_dir}: "
            f"{[p.name for p in llm_candidates]}; pass a single file path.\n",
        )
    return candidates[0]


def _parse_layer_spec(spec: str) -> list[int]:
    """Parse a ``--layers`` spec into a list of layer indices.

    The spec is a comma-separated list where each element is either a
    single integer (``"3"``) or an inclusive range (``"3-7"``).
    Whitespace is stripped. Duplicates are dropped while preserving
    order. Examples::

        "1,3,4,6,7,8"       -> [1, 3, 4, 6, 7, 8]
        "0-6,19-24,34-39"   -> [0, 1, 2, 3, 4, 5, 6, 19, ..., 39]
        "1,3-4,6,7-8"       -> [1, 3, 4, 6, 7, 8]
        "5-5"               -> [5]   (degenerate single-element range)

    Raises :class:`ConfigurationError` on malformed input.
    """
    if not spec or not spec.strip():
        raise ConfigurationError("--layers must be a non-empty list")

    indices: list[int] = []
    seen: set[int] = set()
    for token in spec.split(","):
        piece = token.strip()
        if not piece:
            raise ConfigurationError(
                f"--layers contains an empty entry in {spec!r}"
            )
        if "-" in piece:
            # Inclusive range. Split on the first dash so negative
            # numbers can be expressed if we ever need them (-1 is
            # not a valid layer index for our models, but the parser
            # is forgiving).
            head, sep, tail = piece.partition("-")
            if not head or not tail or sep != "-":
                raise ConfigurationError(
                    f"--layers range {piece!r} is malformed; expected "
                    f"'FROM-TO' with both integers"
                )
            try:
                start = int(head)
                end = int(tail)
            except ValueError as exc:
                raise ConfigurationError(
                    f"--layers range {piece!r} has non-integer bounds: {exc}"
                ) from exc
            if start > end:
                raise ConfigurationError(
                    f"--layers range {piece!r} is reversed "
                    f"({start} > {end})"
                )
            for idx in range(start, end + 1):
                if idx not in seen:
                    seen.add(idx)
                    indices.append(idx)
        else:
            try:
                idx = int(piece)
            except ValueError as exc:
                raise ConfigurationError(
                    f"--layers entry {piece!r} is not an integer: {exc}"
                ) from exc
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
    if not indices:
        raise ConfigurationError(f"--layers {spec!r} produced no indices")
    return indices


def _select_layers(
    manifest_layers: tuple,
    *,
    max_layers: int | None,
    layers_spec: str | None,
) -> tuple:
    """Apply ``--max-layers`` / ``--layers`` to a manifest's layer list.

    Returns the resulting tuple of :class:`LayerDescriptor` objects in
    execution order.

    Both flags are mutually exclusive. With neither flag set the
    full layer list is returned unchanged. Duplicates in ``--layers``
    are de-duplicated while preserving order so ``1,3,3,4`` is
    treated as ``1,3,4``. Ranges are written as ``FROM-TO`` and
    expand inclusively — ``0-6,19-24,34-39`` is equivalent to
    listing every index in those ranges.

    The intent is to support depth-bounded inference (``--max-layers``)
    and ablation-style subset inference (``--layers``) without
    duplicating the index-based ``idx == 0`` / ``idx == last``
    logic that the Qwen2 forwarder would otherwise need. The
    scheduler flags the first and last *selected* layer via
    ``LayerHandles.is_first`` / ``is_last`` so the forwarder can
    decide when to embed tokens and when to apply the final norm +
    LM head.
    """
    if max_layers is not None and layers_spec is not None:
        raise ConfigurationError(
            "--max-layers and --layers are mutually exclusive; "
            "pass only one."
        )
    if max_layers is not None:
        if max_layers <= 0:
            raise ConfigurationError(
                f"--max-layers must be a positive integer, got {max_layers}"
            )
        if max_layers > len(manifest_layers):
            raise ConfigurationError(
                f"--max-layers={max_layers} exceeds the model's "
                f"{len(manifest_layers)} layers"
            )
        return tuple(manifest_layers[:max_layers])
    if layers_spec is not None:
        indices = _parse_layer_spec(layers_spec)
        available = {layer.index for layer in manifest_layers}
        for idx in indices:
            if idx not in available:
                raise ConfigurationError(
                    f"--layers references layer {idx}, which is not in "
                    f"the manifest (have {sorted(available)})"
                )
        layer_map = {layer.index: layer for layer in manifest_layers}
        return tuple(layer_map[idx] for idx in indices)
    return tuple(manifest_layers)


def _load_model_bundle(args, parser: argparse.ArgumentParser) -> dict:
    """Open the model, return a dict with everything both handlers need.

    Centralising this lets ``run`` and ``chat`` share the same
    cache-bump heuristic, vocab-mismatch check, and forwarder setup.
    """
    model_dir, gguf_path, fmt = _resolve_model_paths(args, parser)
    if fmt == "gguf" and gguf_path is None:
        gguf_path = _pick_gguf_file(model_dir, parser)
    print(f"Detected format: {fmt}")

    if fmt == "gguf" and not any(
        (model_dir / f).is_file() for f in ("tokenizer.json", "vocab.json")
    ):
        from flatrun.tokenizer import load_from_gguf_metadata
        print(f"Building tokenizer from GGUF metadata ({gguf_path.name}) ...")
        tokenizer = load_from_gguf_metadata(gguf_path)
    else:
        tok_dir = args.tokenizer or model_dir
        if args.tokenizer is None and fmt == "gguf" and not any(
            (model_dir / f).is_file() for f in ("tokenizer.json", "vocab.json")
        ):
            candidate = (
                Path("/Users/judotens/.lmstudio/models/lmstudio-community/")
                / "Qwen2.5-Coder-7B-Instruct-MLX-4bit"
            )
            if candidate.is_dir():
                tok_dir = candidate
                print(f"  using fallback tokenizer at {tok_dir}")
        tokenizer = auto_load(tok_dir)
    tok_vocab = len(tokenizer.vocab)
    print(f"Tokenizer vocab: {tok_vocab}")
    print(
        f"Chat template: "
        f"{'Qwen2 ChatML' if '<|im_start|>' in tokenizer.chat_template else tokenizer.chat_template[:60] + '...'}"
    )

    cfg = RuntimeConfig(memory=MemoryConfig(cache_bytes=args.cache_mb * 1024 * 1024, probe=None))
    t0 = time.perf_counter()
    loaded = load_huggingface(model_dir, config=cfg)
    if args.cache_mb == 256:
        largest = max(
            (loaded.runtime.get_metadata(k.name).byte_size for k in loaded.runtime.list_tensors()),
            default=0,
        )
        if largest > 0:
            recommended_mb = max(
                256,
                ((largest * 4) + (128 * 1024 * 1024) - 1) // (128 * 1024 * 1024) * 128,
            )
            if recommended_mb > args.cache_mb:
                print(
                    f"  bumping cache from {args.cache_mb} MiB to {recommended_mb} MiB "
                    f"(largest tensor is {largest / 1024 / 1024:.0f} MiB)"
                )
                args.cache_mb = recommended_mb
                loaded.runtime.close()
                cfg = RuntimeConfig(
                    memory=MemoryConfig(cache_bytes=args.cache_mb * 1024 * 1024, probe=None)
                )
                t0 = time.perf_counter()
                loaded = load_huggingface(model_dir, config=cfg)
    print(f"Loaded model in {(time.perf_counter() - t0):.2f} s; layers={loaded.manifest.layer_count}")

    model_vocab: int | None = None
    if fmt == "gguf":
        from flatrun.backend.gguf import GGUFBackend
        be = GGUFBackend(gguf_path)
        be.open()
        try:
            gguf_meta = be.gguf_metadata
        finally:
            be.close()
        model_vocab = len(gguf_meta.get("tokenizer.ggml.tokens", []) or [])
    elif loaded.config is not None and loaded.config.raw is not None:
        model_vocab = int(loaded.config.raw.get("vocab_size", 0)) or None

    if model_vocab is not None and model_vocab != tok_vocab:
        print(
            f"\n*** VOCAB MISMATCH: model={model_vocab}, tokenizer={tok_vocab} ***\n"
            f"    The model's argmax IDs won't map to the tokenizer's vocab.\n"
            f"    Output will be garbage unless you pass --tokenizer pointing\n"
            f"    to a directory whose vocab matches the model.\n",
            file=sys.stderr,
        )

    if fmt == "gguf":
        raw_cfg = _build_config_from_gguf(gguf_path)
        qcfg = Qwen2Config.from_hf_config(raw_cfg)
        qcfg.quant_gguf = args.quant or "Q8_0"
        qcfg.debug_trace = args.debug
    else:
        if loaded.config is None or loaded.config.raw is None:
            parser.exit(1, "No config.json found next to model weights.\n")
        qcfg = Qwen2Config.from_hf_config(loaded.config.raw)
        qcfg.quant_mlx_4bit = fmt == "mlx"
        qcfg.quant_gguf = None
        qcfg.debug_trace = args.debug
    enable_cache = args.dequant_cache == "on"
    selected_layers = _select_layers(
        loaded.manifest.layers,
        max_layers=args.max_layers,
        layers_spec=args.layers,
    )
    if len(selected_layers) != len(loaded.manifest.layers):
        if args.layers is not None:
            print(
                f"Selected layers (custom order): "
                f"{[l.index for l in selected_layers]}"
            )
        else:
            print(
                f"Using first {len(selected_layers)} of "
                f"{len(loaded.manifest.layers)} layers"
            )
    # ``last_index`` for the forwarder is the *original* index of the
    # last selected layer. The scheduler's ``is_last`` flag is the
    # authoritative gate at runtime, but ``last_index`` is the
    # fallback for callers that drive the forwarder outside the
    # scheduler.
    final_layer_index = (
        selected_layers[-1].index
        if len(selected_layers) != len(loaded.manifest.layers)
        else None
    )

    # Special-token IDs come from the tokenizer's added-token table.
    # When the user opts in via ``--debug-include-special`` we pass
    # ``None`` so the per-token debug table shows every token. The
    # analyzer is created only when the user enabled debug-style
    # output (either ``--debug`` or ``--debug-save-analysis``).
    debug_enabled = args.debug or args.debug_save_analysis is not None
    exclude_token_ids = (
        None
        if args.debug_include_special
        else set(int(tid) for tid in getattr(tokenizer, "added_tokens", {}).keys())
    )

    # Build the scheduler first so the analyzer and forwarder can
    # take its ``manager`` — the prediction-evolution analyzer needs
    # to load ``model.norm.weight`` and ``lm_head.weight`` at every
    # layer, not just the last, and those live in the post-layer
    # bookend that the scheduler attaches to the last selected layer.
    scheduler = loaded.runtime.build_scheduler(
        selected_layers,
        pre_layer_names=loaded.manifest.pre_layer,
        post_layer_names=loaded.manifest.post_layer,
    )

    analyzer = None
    if debug_enabled:
        from flatrun.utils.debug import PredictionAnalyzer
        analyzer = PredictionAnalyzer(layer_count=len(selected_layers))

    # The detailed profiler is opt-in: passing ``None`` makes the
    # forwarder's _p helper return a no-op context manager so the
    # hot path has zero overhead.
    profiler = None
    if args.profile_detailed or args.profile_save is not None:
        from flatrun.utils.profiler import Profiler
        profiler = Profiler()

    forwarder = make_qwen2_forwarder(
        qcfg,
        enable_dequant_cache=enable_cache,
        memory_trace=args.memory_trace,
        last_index=final_layer_index,
        tokenizer=tokenizer,
        max_per_token_rows=args.debug_max_token_rows,
        exclude_token_ids=exclude_token_ids,
        analyzer=analyzer,
        manager=scheduler.manager,
        profiler=profiler,
    )
    executor = StreamingExecutor(scheduler, forwarder, kv_cache=KVCache(capacity=4096))

    return {
        "fmt": fmt,
        "gguf_path": gguf_path,
        "tokenizer": tokenizer,
        "loaded": loaded,
        "forwarder": forwarder,
        "executor": executor,
        "qcfg": qcfg,
        "analyzer": analyzer,
        "profiler": profiler,
    }


def _make_sampler(args) -> Sampler:
    if args.no_sample:
        return Sampler(temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0)
    return Sampler(
        temperature=args.temperature,
        top_k=args.sample_top_k,
        top_p=args.sample_top_p,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
        seed=args.seed,
    )


def _generate_continuation(
    bundle: dict,
    args,
    prompt_ids: list[int],
    max_new: int,
    *,
    stream: "typing.Callable[[str], None] | None" = None,
) -> tuple[list[int], np.ndarray | None, list[float]]:
    """Run the forwarder ``max_new`` times and return (ids, last_logits, step_times).

    If ``stream`` is supplied, each newly sampled token is written to
    ``stream(text)`` immediately - typically ``sys.stdout.write`` so
    the user sees the response grow token-by-token, the way Claude
    Code's console streams an answer rather than waiting for the
    full generation. The Spinner from earlier versions is gone;
    progress visibility now comes from the ``[dbg]`` lines on
    stderr (one line per layer, never replaced via ``\\r``) and
    from the in-place token stream on stdout.
    """
    import typing
    executor = bundle["executor"]
    tokenizer = bundle["tokenizer"]
    sampler = _make_sampler(args)
    seen: list[int] = list(prompt_ids)
    step_times: list[float] = []
    logits: np.ndarray | None = None
    next_id = -1
    generated: list[int] = []

    write_token = stream if stream is not None else (lambda _t: None)
    flush_token = (lambda: None) if stream is None else (lambda: stdout_token_flush())

    for nxt in range(max_new):
        ids = prompt_ids + generated if next_id == -1 else prompt_ids + generated
        t0 = time.perf_counter()
        result = executor.step(tokens=ids)
        step_times.append((time.perf_counter() - t0) * 1000)
        if nxt == 0 and args.profile:
            print(f"  initial step: {step_times[-1]:.0f} ms ({len(ids)} tokens)")
        elif args.profile:
            print(f"  step {nxt + 1}: {step_times[-1]:.0f} ms ({len(ids)} tokens)")
        logits = result.last_hidden
        if args.no_sample:
            next_id = int(np.argmax(logits[-1]))
        else:
            next_id = sampler.sample(logits[-1], seen_ids=seen)
        generated.append(next_id)
        seen.append(next_id)
        # Stream the new token to the caller. ``flush_token`` lets the
        # caller batch ``write`` calls (useful for chat where we
        # already print a ``Assistant:`` prefix on the same line).
        write_token(tokenizer.decode([next_id]))
        flush_token()

    return generated, logits, step_times


def stdout_token_flush() -> None:
    sys.stdout.flush()


def cmd_run(args) -> int:
    """One-shot prompt -> continuation (the original behaviour)."""
    parser, _ = _build_argparser()
    if args.max_new is None:
        args.max_new = 1
    bundle = _load_model_bundle(args, parser)
    tokenizer = bundle["tokenizer"]

    if args.no_chat_template:
        prompt_text = args.prompt or ""
    else:
        prompt_text = _assemble_prompt(
            tokenizer,
            messages_json=args.messages_json,
            system=args.system,
            prompt=args.prompt,
        )
    print(f"Prompt:\n{prompt_text}")
    prompt_ids = tokenizer.encode(prompt_text)
    print(f"Prompt tokens: {len(prompt_ids)}")

    sampler = _make_sampler(args)
    print(
        f"  sampling: temperature={sampler.temperature} top_k={sampler.top_k} "
        f"top_p={sampler.top_p} min_p={sampler.min_p} repeat_penalty={sampler.repeat_penalty}"
    )

    print(f"\nRunning initial step with {len(prompt_ids)} prompt tokens ...")
    print("Generated: ", end="", flush=True)
    # ``stream`` writes each new token to stdout as it's sampled, so
    # the user sees the response grow in place rather than waiting
    # for the whole ``--max-new`` budget to finish. The newline is
    # printed in :func:`cmd_run` after the last token, not here, so
    # chat-mode callers can put their own prefix on the same line.
    generated, logits, step_times = _generate_continuation(
        bundle, args, prompt_ids, args.max_new,
        stream=lambda t: print(t, end="", flush=True),
    )
    print()  # newline after the streamed token sequence
    raw_text = tokenizer.decode(generated)
    for nxt, tid in enumerate(generated):
        print(f"  token {nxt + 1}: id={tid} text={tokenizer.decode([tid])!r}")

    if generated:
        thinking, answer = _split_thinking(raw_text)
        if thinking:
            # Layout: ``\n`` then [dim] 'Thinking:' [end] [grey italic]
            # <content> [end]. The dim colour frames just the label;
            # the body is grey italic on the same line. ``_split_thinking``
            # stripped the ``<think>``/``</think>`` markers and
            # ``_flatten_thinking`` collapsed newlines.
            print(f"\n{_C_DIM}Thinking:{_C_END} {_C_GREY}{_C_ITALIC}{thinking}{_C_END}")
        print(f"\nGenerated text: {answer!r}")
    if args.profile and len(step_times) > 1:
        first = step_times[0]
        last = step_times[-1]
        avg = sum(step_times) / len(step_times)
        print(f"\nProfile summary across {len(step_times)} generation steps:")
        print(f"  first step: {first:.0f} ms  (cold cache + dequant)")
        print(f"  last step:  {last:.0f} ms  (warm cache)")
        print(f"  average:    {avg:.0f} ms")
        if first > 0:
            print(f"  speedup from warm cache: {first / max(last, 1):.1f}x")
    if args.top_k > 0 and logits is not None:
        top_k = min(args.top_k, logits.shape[-1])
        top_indices = np.argpartition(logits[-1], -top_k)[-top_k:]
        print(f"\nTop-{top_k} next tokens:")
        for tid in sorted(top_indices, key=lambda i: -logits[-1, i]):
            print(f"  id={tid:6d} logit={logits[-1, tid]:8.2f}  text={tokenizer.decode([tid])!r}")

    _finalize_analyzer(bundle, args)
    _finalize_profiler(bundle, args)
    bundle["loaded"].runtime.close()
    print("Done.")
    return 0


def _finalize_analyzer(bundle: dict, args) -> None:
    """Print the layer analysis summary and persist JSON if requested.

    Called from the bottom of ``cmd_run`` and ``cmd_chat``. The
    analyzer is None when debug-style output was not enabled, so
    this is a no-op in that case.
    """
    analyzer = bundle.get("analyzer")
    if analyzer is not None:
        # Freeze so the autoregressive decode loop's repeated forwards
        # don't accumulate stats (the trigger is the existence of the
        # analyzer; the first forward pass has already populated it).
        analyzer.freeze()
        analyzer.summarize()
        if args.debug_save_analysis is not None:
            path = analyzer.save(args.debug_save_analysis)
            print(f"Layer analysis saved to {path}")


def _finalize_profiler(bundle: dict, args) -> None:
    """Print the per-layer profiler breakdown and persist JSON.

    Called from the bottom of ``cmd_run`` and ``cmd_chat``. The
    profiler is None when ``--profile-detailed`` / ``--profile-save``
    was not requested, so this is a no-op in that case.
    """
    profiler = bundle.get("profiler")
    if profiler is None:
        return
    profiler.freeze()
    profiler.print_per_layer()
    profiler.print_summary()
    if args.profile_save is not None:
        path = profiler.save(args.profile_save)
        print(f"Profiler result saved to {path}")


def cmd_chat(args) -> int:
    """Interactive REPL: 'You: ' prompt, stream the assistant's reply.

    Each turn is rendered through the model's chat template (unless
    ``--no-chat-template`` is set) with the full prior history
    included, so the model sees a real multi-turn conversation. The
    executor is reset between turns so the KV cache cannot drift
    between sessions; the per-turn prefill cost is paid from scratch.
    """
    parser, _ = _build_argparser()
    if args.max_new is None:
        args.max_new = 128
    bundle = _load_model_bundle(args, parser)
    tokenizer = bundle["tokenizer"]
    print(
        f"\nChat mode (max_new={args.max_new}/turn, history={not args.no_history})."
        f"  Type your message; Ctrl-D (EOF) or 'exit' to quit.\n"
    )
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    turn = 0
    while True:
        try:
            user_text = input(f"{_C_USER}You: {_C_USER_END}")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user_text.strip().lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not user_text.strip():
            continue
        messages.append({"role": "user", "content": user_text})

        if args.no_chat_template:
            prompt_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            ) + "\nassistant:"
        else:
            # ``apply_chat_template`` expects the assistant turn to be
            # open; we add an empty assistant slot so the template
            # emits the "now you speak" prefix.
            prompt_text = tokenizer.apply_chat_template(
                [*messages, {"role": "assistant", "content": ""}],
                add_generation_prompt=True,
            )
        prompt_ids = tokenizer.encode(prompt_text)

        if not args.no_history:
            # Render only the latest turn for the model; we pass the
            # full message history to the template but truncate to
            # just the last user turn's encoded text. The chat
            # template does the actual multi-turn formatting.
            pass

        bundle["executor"].kv_cache.reset()
        turn += 1
        t0 = time.perf_counter()
        # Print "Assistant: " in green, then stream the model's
        # tokens onto the next line. The cursor icon lives on the
        # thinking line below the streamed answer (printed by
        # ``_print_thinking``) rather than next to the streamed
        # content - the LiveCursor rewrite-the-cursor-cell-via-\r
        # approach was eating the streamed transcript.
        # The animated yellow cursor and the scrolling thinking
        # content live on a reserved row at the bottom of the
        # terminal (row 999), so they never overwrite the streamed
        # reply on stdout. ``LiveThinkingDisplay`` is the stream
        # callback: it inspects each decoded token, detects the
        # ``<think ...>`` / ``<think ...>`` markers, accumulates the
        # body into a scrolling buffer (kept visible until it
        # exceeds ~20 tokens, then truncated to the tail), and
        # forwards the rest of the stream to stdout unchanged.
        sys.stdout.write(f"{_C_ASSISTANT}Assistant:{_C_ASSISTANT_END}\n")
        sys.stdout.flush()
        with LiveThinkingDisplay() as td:
            generated, _, _ = _generate_continuation(
                bundle, args, prompt_ids, args.max_new,
                stream=td.feed_token,
            )
        print()  # newline after the streamed reply
        dt = time.perf_counter() - t0
        raw_text = tokenizer.decode(generated)
        thinking, reply_text = _split_thinking(raw_text)
        for stop in ("<|im_end|>", "<|endoftext|>", "</s>", "<|end|>"):
            if stop in reply_text:
                reply_text = reply_text.split(stop, 1)[0]
                break
        if thinking:
            # Layout: yellow cursor icon, then dim "Thinking:",
            # then grey italic body. The cursor is the FIRST cell of
            # the thinking line (not the streamed line), so the
            # user sees a clear separation: streamed content on one
            # line, thinking on the next. ``_split_thinking``
            # stripped the ``<think>``/``</think>`` markers and
            # ``_flatten_thinking`` collapsed newlines.
            print(
                f"\n{_C_YELLOW}⠋{_C_END}"
                f"{_C_DIM}Thinking:{_C_END} "
                f"{_C_GREY}{_C_ITALIC}{thinking}{_C_END}"
            )
        # The reply tokens were already streamed to stdout by
        # ``LiveThinkingDisplay.feed_token`` during the ``with``
        # block. Do NOT re-print the reply text here - that would
        # race the streamed output and look like the assistant
        # waited for ``max_new`` before speaking.
        print(f"  ({len(generated)} tokens, {dt:.1f}s, {len(generated) / max(dt, 1e-3):.1f} tok/s)\n")
        messages.append({"role": "assistant", "content": reply_text})

    _finalize_analyzer(bundle, args)
    _finalize_profiler(bundle, args)
    bundle["loaded"].runtime.close()
    return 0


    # 1. Detect format and load tokenizer.
    fmt = _detect_format(model_dir, gguf_path=gguf_path)
    print(f"Detected format: {fmt}")
    # Resolve the on-disk GGUF file once. ``gguf_path`` may have been
    # set from the CLI (file path) or may need to be picked from the
    # directory's first ``.gguf`` file.
    if fmt == "gguf":
        if gguf_path is None:
            candidates = sorted(model_dir.glob("*.gguf"))
            if not candidates:
                print(f"No .gguf file in {model_dir}", file=sys.stderr)
                return 1
            gguf_path = candidates[0]
    tokenizer = None
    # For GGUF dirs without sibling tokenizer files, build the tokenizer
    # directly from the GGUF metadata so the vocab matches the model.
    if fmt == "gguf" and not any(
        (model_dir / f).is_file() for f in ("tokenizer.json", "vocab.json")
    ):
        from flatrun.tokenizer import load_from_gguf_metadata
        print(f"Building tokenizer from GGUF metadata ({gguf_path.name}) ...")
        tokenizer = load_from_gguf_metadata(gguf_path)
    else:
        tok_dir = args.tokenizer or model_dir
        if args.tokenizer is None and fmt == "gguf" and not any(
            (model_dir / f).is_file() for f in ("tokenizer.json", "vocab.json")
        ):
            # Final fallback - the user's MLX 7B is the only sibling
            # tokenizer in their LM Studio download.
            candidate = (
                Path("/Users/judotens/.lmstudio/models/lmstudio-community/")
                / "Qwen2.5-Coder-7B-Instruct-MLX-4bit"
            )
            if candidate.is_dir():
                tok_dir = candidate
                print(f"  using fallback tokenizer at {tok_dir}")
        tokenizer = auto_load(tok_dir)
    tok_vocab = len(tokenizer.vocab)
    print(f"Tokenizer vocab: {tok_vocab}")
    print(f"Chat template: {'Qwen2 ChatML' if '<|im_start|>' in tokenizer.chat_template else tokenizer.chat_template[:60] + '...'}")

    # Pre-load the model early so we can compare vocab sizes. The vocab
    # lives in different places depending on format - GGUF exposes it
    # via the metadata KV table, HF / MLX go through ``Qwen2Config``.
    # Probe the manifest so the default cache can adapt to large
    # models. We need at least 4x the largest single tensor (embed,
    # lm_head) or the cache will evict the embed right after acquiring
    # it, which crashes the forwarder.
    cfg = RuntimeConfig(memory=MemoryConfig(cache_bytes=args.cache_mb * 1024 * 1024, probe=None))
    t0 = time.perf_counter()
    loaded = load_huggingface(model_dir, config=cfg)
    if args.cache_mb == 256:  # user did not override; pick a sensible default
        largest = max(
            (loaded.runtime.get_metadata(k.name).byte_size for k in loaded.runtime.list_tensors()),
            default=0,
        )
        # 4x largest tensor, rounded up to the next 128 MiB, with a
        # 256 MiB floor (matches the explicit default). 4x because the
        # scheduler typically holds pre-layer + post-layer + current
        # layer's tensors concurrently.
        if largest > 0:
            recommended_mb = max(256, ((largest * 4) + (128 * 1024 * 1024) - 1) // (128 * 1024 * 1024) * 128)
            if recommended_mb > args.cache_mb:
                print(
                    f"  bumping cache from {args.cache_mb} MiB to {recommended_mb} MiB "
                    f"(largest tensor is {largest / 1024 / 1024:.0f} MiB)",
                )
                args.cache_mb = recommended_mb
                loaded.runtime.close()
                cfg = RuntimeConfig(memory=MemoryConfig(cache_bytes=args.cache_mb * 1024 * 1024, probe=None))
                t0 = time.perf_counter()
                loaded = load_huggingface(model_dir, config=cfg)
    print(f"Loaded model in {(time.perf_counter() - t0):.2f} s; layers={loaded.manifest.layer_count}")

    model_vocab: int | None = None
    if fmt == "gguf":
        from flatrun.backend.gguf import GGUFBackend
        be = GGUFBackend(gguf_path)
        be.open()
        try:
            gguf_meta = be.gguf_metadata
        finally:
            be.close()
        model_vocab = len(gguf_meta.get("tokenizer.ggml.tokens", []) or [])
    else:
        if loaded.config is not None and loaded.config.raw is not None:
            model_vocab = int(loaded.config.raw.get("vocab_size", 0)) or None

    if model_vocab is not None and model_vocab != tok_vocab:
        print(
            f"\n*** VOCAB MISMATCH: model={model_vocab}, tokenizer={tok_vocab} ***\n"
            f"    The model's argmax IDs won't map to the tokenizer's vocab.\n"
            f"    Output will be garbage unless you pass --tokenizer pointing\n"
            f"    to a directory whose vocab matches the model.\n",
            file=sys.stderr,
        )

    # 2. Render the prompt.
    if args.no_chat_template:
        prompt_text = args.prompt or ""
    else:
        prompt_text = _assemble_prompt(
            tokenizer,
            messages_json=args.messages_json,
            system=args.system,
            prompt=args.prompt,
        )
    print(f"Prompt:\n{prompt_text}")
    prompt_ids = tokenizer.encode(prompt_text)
    print(f"Prompt tokens: {len(prompt_ids)}")

    # 3. Build the Qwen2 forwarder.
    if fmt == "gguf":
        raw_cfg = _build_config_from_gguf(gguf_path)
        qcfg = Qwen2Config.from_hf_config(raw_cfg)
        qcfg.quant_gguf = args.quant or "Q8_0"
        qcfg.debug_trace = args.debug
    else:
        if loaded.config is None or loaded.config.raw is None:
            print("No config.json found next to model weights.", file=sys.stderr)
            return 1
        qcfg = Qwen2Config.from_hf_config(loaded.config.raw)
        qcfg.quant_mlx_4bit = fmt == "mlx"
        qcfg.quant_gguf = None
        qcfg.debug_trace = args.debug
    enable_cache = args.dequant_cache == "on"
    forwarder = make_qwen2_forwarder(qcfg, enable_dequant_cache=enable_cache, memory_trace=args.memory_trace)

    # 4. Stream one prompt + max_new tokens.
    scheduler = loaded.runtime.build_scheduler(
        loaded.manifest.layers,
        pre_layer_names=loaded.manifest.pre_layer,
        post_layer_names=loaded.manifest.post_layer,
    )
    kv = KVCache(capacity=max(128, len(prompt_ids) + args.max_new + 16))
    executor = StreamingExecutor(scheduler, forwarder, kv_cache=kv)

    print(f"\nRunning initial step with {len(prompt_ids)} prompt tokens ...")
    t0 = time.perf_counter()
    result = executor.step(tokens=prompt_ids)
    step_ms = (time.perf_counter() - t0) * 1000
    print(f"  initial step took {step_ms:.1f} ms")
    logits = result.last_hidden
    sampler = Sampler(
        temperature=args.temperature,
        top_k=args.sample_top_k,
        top_p=args.sample_top_p,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
        seed=args.seed,
    )
    if args.no_sample:
        # Force greedy decoding regardless of other settings.
        sampler = Sampler(temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0)
    print(
        f"  sampling: temperature={sampler.temperature} top_k={sampler.top_k} "
        f"top_p={sampler.top_p} min_p={sampler.min_p} repeat_penalty={sampler.repeat_penalty}"
    )
    generated: list[int] = []
    # Tokens the model has already produced - the sampler applies
    # repeat_penalty to these.
    seen: list[int] = list(prompt_ids) + generated
    step_times: list[float] = []
    for nxt in range(args.max_new):
        if args.no_sample:
            next_id = int(np.argmax(logits[-1]))
        else:
            next_id = sampler.sample(logits[-1], seen_ids=seen)
        generated.append(next_id)
        seen.append(next_id)
        next_text = tokenizer.decode([next_id])
        print(f"  token {nxt + 1}: id={next_id} text={next_text!r}")
        if nxt + 1 >= args.max_new:
            break
        # Append the new token to the prompt and step again.
        # FlatRun's executor only does one shot, so we re-run the whole
        # sequence (acceptable for tiny max_new).
        new_ids = prompt_ids + generated
        t0 = time.perf_counter()
        result = executor.step(tokens=new_ids)
        step_ms = (time.perf_counter() - t0) * 1000
        step_times.append(step_ms)
        if args.profile:
            print(f"    step {nxt + 2}: {step_ms:.0f} ms "
                  f"({len(new_ids)} tokens, ~{step_ms / len(new_ids):.1f} ms/tok)")
        logits = result.last_hidden
    if generated:
        print(f"\nGenerated text: {tokenizer.decode(generated)!r}")
    if args.profile and step_times:
        first = step_times[0]
        last = step_times[-1]
        avg = sum(step_times) / len(step_times)
        print(f"\nProfile summary across {len(step_times)} generation steps:")
        print(f"  first step: {first:.0f} ms  (cold cache + dequant)")
        print(f"  last step:  {last:.0f} ms  (warm cache)")
        print(f"  average:    {avg:.0f} ms")
        if first > 0:
            print(f"  speedup from warm cache: {first / max(last, 1):.1f}x")

    # 5. Top-k summary on the final logits.
    if args.top_k > 0 and logits is not None:
        top_k = min(args.top_k, logits.shape[-1])
        top_indices = np.argpartition(logits[-1], -top_k)[-top_k:]
        print(f"\nTop-{top_k} next tokens:")
        for tid in sorted(top_indices, key=lambda i: -logits[-1, i]):
            print(f"  id={tid:6d} logit={logits[-1, tid]:8.2f}  text={tokenizer.decode([tid])!r}")

    loaded.runtime.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
