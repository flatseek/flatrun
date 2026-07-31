"""Tests for the LiveThinkingDisplay streaming state machine.

The display is the chat REPL's animated thinking line. It intercepts
each decoded token from the streaming executor, detects the
``<think ...>`` / ``<think ...>`` markers, accumulates the body into
a scrolling buffer, and forwards the rest of the stream to stdout.

These tests don't drive the animation thread; they verify the
state machine by feeding tokens and checking what would be written
to stdout (the assistant's reply) and what the final ``content``
buffer holds (the reasoning shown on the thinking line).
"""

from __future__ import annotations

import io
import sys

from flatrun.cli import LiveThinkingDisplay


def _capture_stdout(fn):
    """Run ``fn`` with stdout redirected to a buffer, return the buffer text."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn()
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_plain_text_goes_to_stdout() -> None:
    td = LiveThinkingDisplay(stream=io.StringIO())
    out = _capture_stdout(lambda: td.feed_token("hello world"))
    assert out == "hello world"
    assert td._content == ""


def test_think_block_accumulates_and_passes_answer_to_stdout() -> None:
    td = LiveThinkingDisplay(stream=io.StringIO())
    out = _capture_stdout(
        lambda: [
            td.feed_token("<"),
            td.feed_token("think"),
            td.feed_token(">"),
            td.feed_token("reasoning "),
            td.feed_token("step "),
            td.feed_token("by "),
            td.feed_token("step"),
            td.feed_token("</"),
            td.feed_token("think"),
            td.feed_token(">"),
            td.feed_token("final answer"),
        ]
    )
    # Pre-think prefix is empty, so stdout receives only the answer.
    assert out == "final answer"
    # The accumulated body has newlines stripped and whitespace collapsed.
    assert td._content == "reasoning step by step"
    # State machine ended in 'after', so the next token also flows to stdout.
    out2 = _capture_stdout(lambda: td.feed_token(" more"))
    assert out2 == " more"


def test_multiline_thinking_is_collapsed_to_single_line() -> None:
    td = LiveThinkingDisplay(stream=io.StringIO())
    out = _capture_stdout(
        lambda: [
            td.feed_token("<"),
            td.feed_token("think"),
            td.feed_token(">"),
            td.feed_token("line 1\n"),
            td.feed_token("line 2\n"),
            td.feed_token("line 3\n"),
            td.feed_token("</"),
            td.feed_token("think"),
            td.feed_token(">"),
            td.feed_token("answer"),
        ]
    )
    assert out == "answer"
    assert "\n" not in td._content
    assert td._content == "line 1 line 2 line 3"


def test_content_window_truncates_when_exceeding_max_words() -> None:
    td = LiveThinkingDisplay(stream=io.StringIO())
    td.MAX_WORDS = 5
    long_body = " ".join(f"word{i}" for i in range(20))
    out = _capture_stdout(
        lambda: [
            td.feed_token("<"),
            td.feed_token("think"),
            td.feed_token(">"),
            td.feed_token(long_body),
            td.feed_token("</"),
            td.feed_token("think"),
            td.feed_token(">"),
            td.feed_token("done"),
        ]
    )
    assert out == "done"
    # Buffer kept only the last MAX_WORDS words once the buffer
    # exceeded the threshold; shorter content stays untouched.
    assert len(td._content.split(" ")) <= td.MAX_WORDS
    assert td._content.endswith("word19")


def test_content_below_max_words_is_not_truncated() -> None:
    """If the thinking block is short (below the word threshold), the
    full text is kept - the live display never chops a short
    chain-of-thought into a fragment.
    """
    td = LiveThinkingDisplay(stream=io.StringIO())
    td.MAX_WORDS = 15
    short_body = "one two three four five"
    _capture_stdout(
        lambda: [
            td.feed_token("<"),
            td.feed_token("think"),
            td.feed_token(">"),
            td.feed_token(short_body),
            td.feed_token("</"),
            td.feed_token("think"),
            td.feed_token(">"),
            td.feed_token("ok"),
        ]
    )
    assert td._content == short_body


def test_fully_in_buffer_think_block_is_handled_in_one_shot() -> None:
    """A token may contain both the open and close tags."""
    td = LiveThinkingDisplay(stream=io.StringIO())
    out = _capture_stdout(
        lambda: td.feed_token("prefix<think>body</think>suffix")
    )
    # Anything before the open tag streams to stdout; the suffix
    # after the close tag also streams to stdout.
    assert out == "prefixsuffix"
    assert td._content == "body"


def test_non_reasoning_model_streams_plain_text_through() -> None:
    """When the model never emits a ``<think>`` tag, every token
    flows straight to stdout - the Thinking line never gates the
    stream. The placeholder ``...`` body is what the live line
    holds; it's blanked on ``__exit__``.
    """
    td = LiveThinkingDisplay(stream=io.StringIO())
    out = _capture_stdout(
        lambda: [
            td.feed_token("Hello"),
            td.feed_token(", "),
            td.feed_token("world"),
            td.feed_token("!"),
        ]
    )
    assert out == "Hello, world!"
    # No think block ever matched - the content buffer is empty.
    assert td._content == ""
    # State stayed in ``before`` the whole time.
    assert td._state == "before"


def test_rendered_len_advances_during_thinking() -> None:
    """Each token in the thinking state grows ``_rendered_len`` so
    the next append only writes the slice that wasn't on screen.
    No per-token line rewrite, no flicker.
    """
    td = LiveThinkingDisplay(stream=io.StringIO())
    _capture_stdout(
        lambda: [
            td.feed_token("<"),
            td.feed_token("think"),
            td.feed_token(">"),
            td.feed_token("one "),
            td.feed_token("two "),
            td.feed_token("three"),
        ]
    )
    # Body has accumulated; rendered_len should match the body length
    # so the next append won't rewrite the whole line.
    assert td._content == "one two three"
    assert td._rendered_len == len(td._content)


def test_placeholder_drops_when_first_real_content_arrives() -> None:
    """After ``_activate`` paints the ``...`` placeholder, the very
    first real-content write must replace the placeholder entirely
    - the appender would otherwise leave ``...real content`` on
    screen.
    """
    td = LiveThinkingDisplay(stream=io.StringIO())
    # Pre-activate the line the way the animation thread would.
    td._activate()
    assert td._placeholder_on_screen is True
    assert td._content == ""
    # Drive a token that opens a think block AND has body content.
    _capture_stdout(
        lambda: [
            td.feed_token("<"),
            td.feed_token("think"),
            td.feed_token(">"),
            td.feed_token("reasoning"),
        ]
    )
    # After the first real-content paint, the placeholder flag is
    # cleared. The body has length 9 ("reasoning") and rendered_len
    # matches it - a full rewrite happened, not an append over the
    # placeholder.
    assert td._content == "reasoning"
    assert td._placeholder_on_screen is False
    assert td._rendered_len == len(td._content)
