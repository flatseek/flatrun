"""Tests for the sampling module."""

from __future__ import annotations

import numpy as np
import pytest

from flatrun.model.sampling import (
    Sampler,
    sample_min_p,
    sample_repeat_penalty,
    sample_temperature,
    sample_token,
    sample_top_k,
    sample_top_p,
    softmax,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_softmax_sums_to_one() -> None:
    logits = np.array([1.0, 2.0, 3.0, 4.0])
    p = softmax(logits)
    np.testing.assert_allclose(p.sum(), 1.0, rtol=1e-6)


def test_sample_temperature_scales_logit() -> None:
    out = sample_temperature(np.array([4.0, 2.0, 0.0]), 0.5)
    np.testing.assert_allclose(out, np.array([8.0, 4.0, 0.0]))


def test_sample_temperature_rejects_zero() -> None:
    with pytest.raises(ValueError, match="temperature"):
        sample_temperature(np.array([1.0]), 0.0)


def test_sample_top_k_keeps_only_top_k() -> None:
    out = sample_top_k(np.array([1.0, 5.0, 2.0, 4.0, 3.0]), top_k=2)
    # Only the two largest (5.0, 4.0) survive.
    assert out[1] == 5.0
    assert out[3] == 4.0
    assert out[0] == float("-inf")
    assert out[2] == float("-inf")
    assert out[4] == float("-inf")


def test_sample_top_k_zero_disables() -> None:
    out = sample_top_k(np.array([1.0, 2.0, 3.0]), 0)
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))


def test_sample_top_p_keeps_nucleus() -> None:
    # When one token dominates, the nucleus is just that one token.
    out = sample_top_p(np.array([0.0, 10.0, 0.0, 0.0]), top_p=0.9)
    assert out[1] == 10.0
    for i in (0, 2, 3):
        assert out[i] == float("-inf")


def test_sample_top_p_rejects_bad_value() -> None:
    with pytest.raises(ValueError, match="top_p"):
        sample_top_p(np.array([0.0, 1.0]), 1.5)


def test_sample_min_p_drops_low_prob_tokens() -> None:
    # One token dominates; min_p=0.1 should drop the small ones.
    out = sample_min_p(np.array([0.0, 10.0, -10.0, -10.0]), min_p=0.1)
    assert out[1] == 10.0
    for i in (0, 2, 3):
        assert out[i] == float("-inf")


def test_sample_repeat_penalty_divide_positive() -> None:
    out = sample_repeat_penalty(np.array([4.0, 2.0, -1.0]), [0], penalty=2.0)
    np.testing.assert_allclose(out, np.array([2.0, 2.0, -1.0]))


def test_sample_repeat_penalty_multiply_negative() -> None:
    out = sample_repeat_penalty(np.array([4.0, 2.0, -1.0]), [2], penalty=2.0)
    np.testing.assert_allclose(out, np.array([4.0, 2.0, -2.0]))


def test_sample_repeat_penalty_noop_when_one() -> None:
    out = sample_repeat_penalty(np.array([4.0, 2.0, -1.0]), [0, 1, 2], penalty=1.0)
    np.testing.assert_array_equal(out, np.array([4.0, 2.0, -1.0]))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def test_sample_token_returns_valid_index() -> None:
    logits = np.zeros(100)
    logits[42] = 1.0
    rng = np.random.default_rng(0)
    for _ in range(20):
        tid = sample_token(logits, temperature=0.01, rng=rng)
        assert tid == 42  # near-greedy


def test_sample_token_diversity() -> None:
    # A flat distribution should sample many distinct tokens across many calls.
    rng = np.random.default_rng(0)
    logits = np.ones(50)
    seen = set()
    for _ in range(100):
        seen.add(sample_token(logits, temperature=1.0, rng=rng))
    assert len(seen) > 10


def test_sampler_defaults_match_qwen_recipe() -> None:
    """The Sampler defaults match the requested Qwen2.5-Coder recipe."""
    s = Sampler()
    assert s.temperature == 0.11
    assert s.top_k == 20
    assert s.top_p == 0.59
    assert s.min_p == 0.05
    assert s.repeat_penalty == 1.1


def test_sampler_with_seed_is_deterministic() -> None:
    logits = np.random.default_rng(0).standard_normal(1000)
    s = Sampler(seed=42)
    out_a = [s.sample(logits) for _ in range(5)]
    out_b = [s.sample(logits) for _ in range(5)]
    assert out_a == out_b


def test_sampler_applies_repeat_penalty() -> None:
    """The same token shouldn't dominate forever under repeat_penalty."""
    s = Sampler(temperature=1.0, top_k=50, top_p=1.0, min_p=0.0, repeat_penalty=10.0, seed=0)
    logits = np.zeros(20)
    logits[5] = 5.0  # one strong token
    seen: list[int] = [5]
    # After the first call, 5's logit should drop dramatically.
    out = s.sample(logits, seen_ids=seen)
    # The sampled token won't be 5 (penalty drops its logit below the
    # -inf threshold once seen).
    assert out != 5
