"""Tests for the chat-mode thinking extractor.

The CLI splits the chain-of-thought off the visible reply so the
REPL can show ``(thinking: N chars)`` and the assistant's actual
answer as separate blocks. These cases cover the patterns the
extractor has to recognise.
"""

from flatrun.cli import _split_thinking


def test_no_thinking_returns_none() -> None:
    thinking, answer = _split_thinking("plain answer")
    assert thinking is None
    assert answer == "plain answer"


def test_extracts_qwen3_think_block() -> None:
    text = "<think>reasoning goes here</think>actual answer"
    thinking, answer = _split_thinking(text)
    assert thinking == "reasoning goes here"
    assert answer == "actual answer"


def test_strips_thinking_then_end_marker() -> None:
    text = "<think>thinking</think>\n\nfinal"
    thinking, answer = _split_thinking(text)
    assert thinking == "thinking"
    assert answer.rstrip() == "final"
    # The CLI truncates at <|im_end|> separately; _split_thinking
    # just splits the thinking block and lstrip()s whitespace,
    # leaving the marker for the caller to chop.


def test_unterminated_thinking_becomes_thinking() -> None:
    text = "<think>never closes..."
    thinking, answer = _split_thinking(text)
    assert thinking == "never closes..."
    assert answer == ""


def test_thinking_in_middle_of_text() -> None:
    text = "prefix<think>reasoning</think>suffix"
    thinking, answer = _split_thinking(text)
    assert thinking == "reasoning"
    assert answer == "prefixsuffix"


def test_empty_thinking_block_is_dropped() -> None:
    thinking, answer = _split_thinking("<think></think>actual")
    assert thinking is None
    assert answer == "actual"