"""Token sampling for autoregressive generation.

Pure NumPy implementation of the standard generation-time techniques:

* :func:`sample_temperature` - divide logits by ``temperature`` then
  rescale.
* :func:`sample_top_k` - keep only the top-``k`` logit values, zero
  out the rest.
* :func:`sample_top_p` (nucleus) - keep the smallest set of tokens
  whose cumulative probability exceeds ``p``.
* :func:`sample_min_p` - keep tokens whose probability is at least
  ``min_p * max_probability``.
* :func:`sample_repeat_penalty` - divide positive logits / multiply
  negative logits for tokens seen in the prompt by ``repeat_penalty``.
* :func:`sample_token` - the full pipeline: temperature -> top_k ->
  top_p -> min_p -> categorical sample.

The :class:`Sampler` dataclass bundles the typical parameters and
exposes a single :meth:`Sampler.sample` entry point so call sites don't
have to plumb five flags through.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sample_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by ``temperature`` (clamped to a positive value)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / float(temperature)


def sample_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
    """Zero out everything outside the top-``k`` largest logits."""
    if top_k <= 0:
        return logits
    k = min(int(top_k), logits.shape[-1])
    # ``argpartition`` is O(n) and works without sorting the whole array.
    threshold = np.partition(logits, -k)[-k]
    return np.where(logits >= threshold, logits, float("-inf"))


def sample_top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
    """Nucleus sampling: keep the smallest token set with cumsum >= ``top_p``."""
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    # Shift to [-max, 0] for numerical stability before softmax.
    shifted = logits - logits.max()
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    sorted_idx = np.argsort(-probs)  # descending
    sorted_probs = probs[sorted_idx]
    cum = np.cumsum(sorted_probs)
    keep = cum <= top_p
    # Always keep at least the top token, even if its prob > top_p.
    keep[0] = True
    mask = np.zeros_like(probs, dtype=bool)
    mask[sorted_idx[keep]] = True
    return np.where(mask, logits, float("-inf"))


def sample_min_p(logits: np.ndarray, min_p: float) -> np.ndarray:
    """Drop tokens whose probability is below ``min_p * max_probability``."""
    if not 0.0 <= min_p <= 1.0:
        raise ValueError(f"min_p must be in [0, 1], got {min_p}")
    if min_p == 0.0:
        return logits
    shifted = logits - logits.max()
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    max_p = probs.max()
    mask = probs >= min_p * max_p
    return np.where(mask, logits, float("-inf"))


from collections.abc import Sequence


def sample_repeat_penalty(
    logits: np.ndarray,
    seen_ids: Sequence[int],
    penalty: float,
) -> np.ndarray:
    """Apply repetition penalty in-place style (returns a new array).

    For each token in ``seen_ids``:
    * if its current logit is positive, divide by ``penalty``;
    * if negative, multiply by ``penalty``.

    ``penalty == 1.0`` is a no-op; ``penalty < 1.0`` would *encourage*
    repetition (we don't recommend that).
    """
    if penalty == 1.0 or not seen_ids:
        return logits
    out = logits.copy()
    for tid in seen_ids:
        if 0 <= tid < out.shape[-1]:
            v = out[tid]
            out[tid] = v / penalty if v > 0 else v * penalty
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max()
    exps = np.exp(shifted)
    return exps / exps.sum()


def sample_token(
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    seen_ids: Sequence[int] | None = None,
    repeat_penalty: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one token from ``logits`` with the full filtering pipeline.

    The order is: repeat_penalty -> temperature -> top_k -> top_p ->
    min_p -> categorical sample.
    """
    if seen_ids is not None and repeat_penalty != 1.0:
        logits = sample_repeat_penalty(logits, seen_ids, repeat_penalty)
    if temperature != 1.0:
        logits = sample_temperature(logits, temperature)
    if top_k > 0:
        logits = sample_top_k(logits, top_k)
    if top_p < 1.0:
        logits = sample_top_p(logits, top_p)
    if min_p > 0.0:
        logits = sample_min_p(logits, min_p)
    probs = softmax(logits)
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice(probs.shape[-1], p=probs))


@dataclass(slots=True)
class Sampler:
    """Bundle of sampling parameters + a single ``.sample()`` entry point.

    Defaults match the Qwen2.5-Coder recommended recipe:

    * temperature = 0.11  (near-greedy, very low)
    * top_k = 20
    * top_p = 0.59
    * min_p = 0.05
    * repeat_penalty = 1.1
    """

    temperature: float = 0.11
    top_k: int = 20
    top_p: float = 0.59
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    seed: int | None = None

    def sample(
        self,
        logits: np.ndarray,
        *,
        seen_ids: Sequence[int] | None = None,
    ) -> int:
        """Sample one token, applying the configured pipeline."""
        rng = None
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
        return sample_token(
            logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            seen_ids=seen_ids,
            repeat_penalty=self.repeat_penalty,
            rng=rng,
        )


__all__ = [
    "Sampler",
    "sample_min_p",
    "sample_repeat_penalty",
    "sample_temperature",
    "sample_token",
    "sample_top_k",
    "sample_top_p",
    "softmax",
]
