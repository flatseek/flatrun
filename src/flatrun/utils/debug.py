"""Generic per-token debug metrics and layer analysis for transformer
hidden states.

Architecture-agnostic helpers that operate on any ``(seq_len,
hidden_size)`` hidden-state tensor. They are wired into the Qwen2 /
Llama / Gemma / Qwen3.5 forwarders by importing from this module
and calling the table printer after each decoder block.

The columns reflect three confidence levels:

* **Always computable** — ``norm``, ``delta``, ``stable``. These are
  derived from the hidden state and the previous layer's hidden state
  alone. Combined, they describe how the per-token representation is
  moving through the residual stream.
* **Soft-computable** — ``influence``. The decoder block computes
  attention via ``einsum`` but does not expose the matrix. Until we
  thread that through, we report a *norm-based proxy* (ratio of the
  token's L2 norm to the mean L2 norm of the row set). The proxy is
  cheap and tracks a noisy "is this token carrying more residual
  energy than its neighbours". Replace with attention-sums-to-token
  once the forwarder exposes the attention matrix. See the
  ``# TODO`` in :func:`per_token_metrics`.
* **Last-layer only** — ``entropy`` and ``conf``. Both come from the
  softmax of the LM-head logits. Shown as ``-`` for non-final layers
  so the column is always present and the user can see at a glance
  which layer is the last.

The output is sorted by ``norm`` (descending) when attention is not
yet available, and by ``influence`` once the attention matrix is
exposed. The user requested "sort by influence when available,
otherwise by norm" — the current default of norm reflects that the
attention column is still a proxy.

The :class:`LayerAnalyzer` collects per-layer statistics across the
forward pass and produces a post-inference summary with a composite
"layer activity" score, the most active / most stable layers, a
suggested early-exit layer, and a suggested layer subset. The score
is a heuristic:

* 0.4 × mean ``delta_norm`` (residual stream displacement)
* 0.3 × ``delta_pct`` (relative magnitude change)
* 0.3 × ``(1 - stable)`` (angular distance from previous layer)

The weights favour "how much did the data move" over "did the
direction change", which lines up with the empirical observation
that mid-network layers tend to have low-stability but high-delta
behaviour (mixing-heavy) and vice versa for the final layers
(representation-fixing). The exact weights are exposed as
:class:`LayerAnalyzer` class attributes so they can be tuned per
model without editing the call sites.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def per_token_metrics(
    h: np.ndarray,
    prev_h: np.ndarray | None,
    *,
    tokens: np.ndarray | None = None,
    logits: np.ndarray | None = None,
) -> dict[str, np.ndarray | None]:
    """Compute per-token metrics for a hidden state.

    Parameters
    ----------
    h : np.ndarray
        Current hidden state, shape ``(seq_len, hidden_size)``.
    prev_h : np.ndarray | None
        Previous layer's hidden state, same shape as ``h``. ``None``
        on the very first layer.
    tokens : np.ndarray | None
        Token IDs that produced ``h`` (length ``seq_len``). Stored
        back so the printer can pair IDs with their decoded names.
    logits : np.ndarray | None
        LM-head logits, shape ``(seq_len, vocab_size)``. ``None``
        for non-final layers; the entropy / confidence columns are
        then left as ``None`` in the returned dict.

    Returns
    -------
    dict
        ``{norms, deltas, stables, influences, entropies, confs,
        tokens}`` — each value is either a ``np.ndarray`` of shape
        ``(seq_len,)`` or ``None`` when not available.
    """
    seq_len = int(h.shape[0])
    norms = np.linalg.norm(h, axis=-1)

    deltas = np.full(seq_len, np.nan, dtype=np.float32)
    stables = np.full(seq_len, np.nan, dtype=np.float32)
    if prev_h is not None and prev_h.shape == h.shape:
        diffs = h - prev_h
        deltas = np.linalg.norm(diffs, axis=-1)
        norm_prev = np.linalg.norm(prev_h, axis=-1)
        denom = norms * norm_prev + 1e-9
        # cosine(hidden_now, hidden_prev) per token. Fail-soft if
        # either side is zero (would otherwise NaN).
        stables = np.einsum("ij,ij->i", h, prev_h) / denom
        # Replace any residual NaN with 0.0 so the table layout is
        # clean — a zero-norm vector is "no signal" and reading 0.0
        # is less alarming than NaN.
        stables = np.where(np.isnan(stables), 0.0, stables)

    # Influence: norm ratio over the row mean. The proxy is a noisy
    # but always-on stand-in for "this token carries more residual
    # energy than its neighbours". A value > 1.0 means above-mean
    # magnitude, < 1.0 means below-mean.
    #
    # TODO: replace with the actual sum of attention weights
    # directed at this token once the forwarder exposes the
    # attention matrix. The decoder block currently computes
    # ``attn = softmax(Q·K^T)`` and consumes it without storing it.
    # Plumbing that through requires capturing the matrix before
    # the ``_softmax`` call (or hooking via a return tuple).
    influences = norms / (norms.mean() + 1e-9)

    entropies: np.ndarray | None = None
    confs: np.ndarray | None = None
    if logits is not None:
        # Stable softmax over the vocab axis.
        logits_max = logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        entropies = -np.sum(probs * np.log(probs + 1e-9), axis=-1)
        confs = probs.max(axis=-1)

    return {
        "norms": norms,
        "deltas": deltas,
        "stables": stables,
        "influences": influences,
        "entropies": entropies,
        "confs": confs,
        "tokens": tokens,
    }


def print_per_token_table(
    layer_idx: int,
    metrics: dict[str, np.ndarray | None],
    *,
    tokenizer: Any | None = None,
    max_rows: int = 16,
    sort_by: str = "norm",
    stream: Any = None,
    exclude_ids: set[int] | None = None,
) -> None:
    """Write a per-token metrics table to ``stream``.

    Parameters
    ----------
    layer_idx : int
        Layer number, used in the leading ``[dbg Lxx]`` header.
    metrics : dict
        Output of :func:`per_token_metrics`.
    tokenizer : optional
        Tokenizer with a ``decode([int]) -> str`` method. When
        provided, each row's decoded token is shown (truncated to
        14 chars). Without it, the ``decode`` column shows ``-``.
    max_rows : int
        Maximum number of rows to print. Sequences longer than
        ``max_rows`` are truncated to the top-N by the sort key.
    sort_by : str
        Either ``"norm"`` or ``"influence"``. The key is read from
        ``metrics``; missing keys fall back to ``metrics["norms"]``
        so a typo never crashes the run.
    stream : file-like
        Defaults to ``sys.stderr``. The signature exists so a
        test can swap in an ``io.StringIO``.
    exclude_ids : set[int] | None
        Token IDs to omit from the table. Pass the tokenizer's
        ``added_tokens.keys()`` to suppress BOS / EOS / chat
        template markers so the table focuses on content tokens.
        When ``None`` (the default) no filtering is applied.
    """
    if stream is None:
        stream = sys.stderr

    norms = metrics["norms"]
    deltas = metrics["deltas"]
    stables = metrics["stables"]
    influences = metrics["influences"]
    entropies = metrics["entropies"]
    confs = metrics["confs"]
    tokens = metrics["tokens"]
    seq_len = int(norms.shape[0])

    # Per-metric 1-indexed descending ranks. NaN values get rank
    # 0 (rendered as ``-``) so the very first layer - where
    # deltas and stables are undefined because there is no prev_h
    # - still produces a parseable table. The double ``argsort``
    # idiom is O(N log N) and the N is at most ``max_rows`` once
    # we filter.
    def _rank_desc(arr: np.ndarray) -> np.ndarray:
        nan_mask = np.isnan(arr)
        # Replace NaN with -inf so they sort to the END (lowest
        # "best" rank). The mask is reapplied at the end so the
        # caller can render NaN as ``-``.
        safe = np.where(nan_mask, -np.inf, arr)
        ranks = np.argsort(np.argsort(-safe)).astype(np.int64) + 1
        ranks = np.where(nan_mask, 0, ranks)
        return ranks

    rank_by_norm = _rank_desc(norms)
    rank_by_delta = _rank_desc(deltas)
    rank_by_stable = _rank_desc(stables)

    sort_key = (
        metrics[sort_by]
        if sort_by in {"norm", "influence"} and sort_by in metrics and metrics[sort_by] is not None
        else norms
    )
    if sort_by == "influence" and (sort_key is influences and influences is None):
        sort_key = norms
    sorted_idx = np.argsort(-sort_key)

    # Filter out special tokens before truncating. The header
    # reports both the hidden count and the visible count so the
    # user sees at a glance how many tokens were suppressed.
    excluded_count = 0
    if exclude_ids is not None and tokens is not None:
        # Build a boolean mask aligned with sorted_idx.
        keep_mask = np.array(
            [int(tokens[i]) not in exclude_ids for i in sorted_idx],
            dtype=bool,
        )
        excluded_count = int(len(sorted_idx) - int(keep_mask.sum()))
        sorted_idx = sorted_idx[keep_mask]

    if len(sorted_idx) > max_rows:
        # Keep the top-N. If the user really wants the tail, they
        # can flip --max-token-rows or write a custom hook.
        sorted_idx = sorted_idx[:max_rows]

    # Pre-decode the tokens we are about to print. Cap the column
    # at 14 chars so the table stays readable for languages whose
    # decoded form is wide (CJK, emoji, ...) or for tokens that
    # decode to a tokeniser-specific multi-byte marker.
    decode_map: dict[int, str] = {}
    if tokens is not None and tokenizer is not None:
        for idx in sorted_idx:
            tid = int(tokens[idx])
            if tid not in decode_map:
                try:
                    decoded = tokenizer.decode([tid])
                except Exception:
                    decoded = ""
                # Strip newlines so each row stays on one line.
                decoded = decoded.replace("\n", "\\n").replace("\r", "\\r")
                decode_map[tid] = repr(decoded)[:14]

    # Header. The "(sorted by ...)" prefix tells the user at a
    # glance whether they're reading by norm or by influence — the
    # two orderings disagree once the attention matrix replaces the
    # proxy.
    sort_label = "influence" if sort_by == "influence" and influences is not None else "norm"
    shown = len(sorted_idx)
    if excluded_count > 0:
        considered = seq_len - excluded_count
        stream.write(
            f"[dbg L{layer_idx:2d}] per-token (sorted by {sort_label} DESC, "
            f"showing {shown}/{considered}; excluded {excluded_count} special "
            f"of {seq_len}):\n"
        )
    else:
        stream.write(
            f"[dbg L{layer_idx:2d}] per-token (sorted by {sort_label} DESC, "
            f"showing {shown}/{seq_len}):\n"
        )
    # Column header — fixed widths keep rows aligned. The right-edge
    # numeric columns use direct fp formatting so the alignment is
    # byte-stable regardless of the layer's hidden-size.
    stream.write(
        f"  {'rank':>4}  {'id':>6}  {'decode':<14}  {'norm':>8}  "
        f"{'delta':>8}  {'stable':>6}  {'infl':>6}  {'H':>7}  {'p_max':>7}  "
        f"{'rank_delta':>10}  {'rank_stable':>11}\n"
    )

    for rank, idx in enumerate(sorted_idx, 1):
        tid = int(tokens[idx]) if tokens is not None else -1
        decode = decode_map.get(tid, "-")
        delta_val = deltas[idx]
        stable_val = stables[idx]
        infl_val = influences[idx]
        norm_val = norms[idx]
        ent_val = entropies[idx] if entropies is not None else None
        conf_val = confs[idx] if confs is not None else None

        # NaN deltas/stables happen on the very first layer (no
        # prev_h). Render as "-" so the column is still parseable.
        delta_str = (
            f"{delta_val:>8.3f}"
            if not (isinstance(delta_val, float) and np.isnan(delta_val))
            else f"{'-':>8}"
        )
        stable_str = (
            f"{stable_val:>6.3f}"
            if not (isinstance(stable_val, float) and np.isnan(stable_val))
            else f"{'-':>6}"
        )
        ent_str = f"{ent_val:>7.3f}" if ent_val is not None else f"{'-':>7}"
        conf_str = f"{conf_val:>7.3f}" if conf_val is not None else f"{'-':>7}"
        delta_rank = int(rank_by_delta[idx])
        stable_rank = int(rank_by_stable[idx])
        delta_rank_str = (
            f"{delta_rank:>10}" if delta_rank > 0 else f"{'-':>10}"
        )
        stable_rank_str = (
            f"{stable_rank:>11}" if stable_rank > 0 else f"{'-':>11}"
        )

        stream.write(
            f"  {rank:>4}  {tid:>6}  {decode:<14}  {norm_val:>8.3f}  "
            f"{delta_str}  {stable_str}  {infl_val:>6.2f}  {ent_str}  {conf_str}  "
            f"{delta_rank_str}  {stable_rank_str}\n"
        )

    stream.flush()


__all__ = ["per_token_metrics", "print_per_token_table", "PredictionAnalyzer"]


# ---------------------------------------------------------------------------
# Prediction Evolution Analyzer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PredictionStats:
    """Per-layer prediction output.

    Captures the model's prediction at the *last* sequence position
    (the next-token distribution) after applying the final norm +
    LM head to the layer's hidden state. The fields are recorded
    once per layer from the prompt prefill pass; later decode
    steps are skipped because the analyzer deduplicates by
    ``layer``.
    """

    layer: int
    top1_id: int
    top1_text: str
    top1_prob: float
    top5_ids: list[int]
    top5_probs: list[float]
    top5_texts: list[str]
    entropy: float
    margin: float
    confidence: float
    prediction_changed: bool
    prev_top1_id: int
    prev_top1_text: str
    delta_confidence: float


class PredictionAnalyzer:
    """Track per-layer prediction evolution.

    The forwarder applies the final RMSNorm + LM head to the
    layer's hidden state at every layer, producing a ``(seq_len,
    vocab)`` logits tensor. The analyzer works on the *last*
    position only (the next-token prediction) and computes:

    * top-5 token IDs + probabilities
    * softmax entropy
    * margin (top1_prob - top2_prob)
    * confidence (top1_prob)
    * prediction_changed (vs the previous recorded layer)
    * delta_confidence

    The post-inference summary covers:

    * Layer-by-layer prediction evolution table
    * Confidence growth across all layers
    * Most Influential Layers (top-k by delta_confidence)
    * Prediction Changes (transitions of the top-1 token)
    * Prediction Stabilization Layer (first layer where top1
      stops changing and confidence keeps increasing)
    * Suggested Early Exit (first layer where top1 is stable,
      confidence >= 95% of final, entropy and margin are stable)
    * JSON dump of every per-layer record for cross-prompt
      research

    Why prediction-based rather than hidden-state-based? The hidden
    state can swing wildly (large norm change, low cosine) without
    the prediction changing at all — that's normal for
    representation-fixing layers near the end. Conversely, a
    hidden state can be nearly identical to the previous layer
    yet the prediction can sharpen (a small rotation in the
    direction of the right answer). The prediction-centric view
    measures what the user actually cares about: "did this layer
    help the model decide which token to print next?".
    """

    # Threshold for "confidence is close enough to the final
    # confidence to declare the prediction stable". Expressed as
    # a fraction of the final-layer confidence.
    early_exit_ratio: float = 0.95

    # Number of layers to skip before the early-exit search is
    # allowed to fire. The first few layers are always active
    # regardless of confidence.
    early_exit_skip: int = 3

    # Tolerance for entropy / margin stability. The candidate
    # layer's entropy (or margin) must be within this fraction of
    # the final layer's entropy (or margin) for the early-exit
    # criterion to count it as "stable".
    early_exit_tolerance: float = 0.05

    # Number of top influential layers to surface.
    top_k: int = 5

    def __init__(self, layer_count: int) -> None:
        self._layer_count = layer_count
        self._layers: list[PredictionStats] = []
        self._seen: set[int] = set()
        self._frozen: bool = False

    def record(
        self,
        layer_idx: int,
        logits: np.ndarray,
        *,
        tokens: np.ndarray | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """Record predictions for one layer.

        ``logits`` is the full ``(seq_len, vocab)`` tensor from
        applying the final norm + LM head to the layer's hidden
        state. The analyzer reads the last row (the next-token
        position) and computes summary statistics.

        Only the first call per ``layer_idx`` is kept; subsequent
        calls (from the autoregressive decode loop) are silently
        dropped. The first forward pass is the prompt prefill;
        that is the most informative single window for tracking
        prediction evolution because the forwarder sees the full
        sequence.
        """
        if self._frozen or layer_idx in self._seen:
            return
        self._seen.add(layer_idx)

        # Last position = the next-token prediction. The forwarder
        # runs the full sequence through every layer so the shape
        # is (seq_len, vocab); we want (vocab,).
        last_logits = np.asarray(logits[-1], dtype=np.float32)
        # Numerically stable softmax.
        logits_max = float(last_logits.max())
        exp_logits = np.exp(last_logits - logits_max)
        probs = exp_logits / exp_logits.sum()

        # Top-5 indices, sorted by probability descending.
        top5_idx = np.argpartition(-probs, min(5, len(probs) - 1))[:5]
        top5_idx = top5_idx[np.argsort(-probs[top5_idx])]
        top5_probs = probs[top5_idx].astype(np.float64)

        top1_id = int(top5_idx[0])
        top1_prob = float(top5_probs[0])
        top2_prob = float(top5_probs[1]) if len(top5_probs) > 1 else 0.0
        margin = top1_prob - top2_prob

        # Entropy of the full softmax distribution. Log base e —
        # the user-facing report formats it via the same scale.
        entropy = -float(np.sum(probs * np.log(probs + 1e-9)))

        # Decode top-1 and top-5 for human-readable output.
        if tokenizer is not None:
            try:
                top1_text = tokenizer.decode([top1_id])
            except Exception:
                top1_text = ""
            top5_texts: list[str] = []
            for tid in top5_idx:
                try:
                    decoded = tokenizer.decode([int(tid)])
                except Exception:
                    decoded = ""
                # Strip newlines so each row stays on one line.
                top5_texts.append(decoded.replace("\n", "\\n").replace("\r", "\\r"))
        else:
            top1_text = ""
            top5_texts = []

        # Delta vs previous layer.
        if self._layers:
            prev_stats = self._layers[-1]
            prev_top1_id = prev_stats.top1_id
            prev_top1_text = prev_stats.top1_text
            prediction_changed = prev_top1_id != top1_id
            delta_confidence = top1_prob - prev_stats.confidence
        else:
            prev_top1_id = -1
            prev_top1_text = ""
            prediction_changed = False
            delta_confidence = 0.0

        self._layers.append(
            PredictionStats(
                layer=int(layer_idx),
                top1_id=top1_id,
                top1_text=top1_text,
                top1_prob=top1_prob,
                top5_ids=[int(i) for i in top5_idx],
                top5_probs=[float(p) for p in top5_probs],
                top5_texts=top5_texts,
                entropy=entropy,
                margin=margin,
                confidence=top1_prob,
                prediction_changed=bool(prediction_changed),
                prev_top1_id=prev_top1_id,
                prev_top1_text=prev_top1_text,
                delta_confidence=float(delta_confidence),
            )
        )

    def freeze(self) -> None:
        """Stop accepting new ``record`` calls.

        Called by the CLI after the first forward pass completes
        so the autoregressive decode loop doesn't churn the
        analyzer. Safe to call more than once.
        """
        self._frozen = True

    def layer_count(self) -> int:
        return len(self._layers)

    def _ordered(self) -> list[PredictionStats]:
        """Return recorded layers in increasing layer-index order."""
        return sorted(self._layers, key=lambda l: l.layer)

    def _stabilization_layer(
        self, layers: list[PredictionStats]
    ) -> int | None:
        """First layer where top1 is stable AND confidence keeps
        increasing to the end.

        "Top1 stable" = top1_id at this layer equals top1_id at
        every subsequent layer. "Confidence keeps increasing" =
        the confidence at this layer is the minimum of all
        confidence values from this layer onwards.
        """
        if not layers:
            return None
        n = len(layers)
        for i, layer in enumerate(layers):
            top1_id = layer.top1_id
            top1_stable = all(layers[j].top1_id == top1_id for j in range(i, n))
            if not top1_stable:
                continue
            confs = [layers[j].confidence for j in range(i, n)]
            if not confs:
                continue
            if min(confs) == layer.confidence:
                return layer.layer
        return None

    def _most_influential(
        self, layers: list[PredictionStats]
    ) -> list[PredictionStats]:
        """Top-k layers by delta_confidence (descending)."""
        return sorted(
            layers,
            key=lambda l: (-l.delta_confidence, l.layer),
        )[: self.top_k]

    def _prediction_changes(
        self, layers: list[PredictionStats]
    ) -> list[PredictionStats]:
        """Layers whose top1 changed from the previous recorded layer."""
        return [l for l in layers if l.prediction_changed]

    def _suggested_early_exit(
        self, layers: list[PredictionStats]
    ) -> dict[str, Any] | None:
        """Find the first layer where all four criteria are met:

        * Top1 doesn't change after this layer (read forward-only)
        * Confidence >= ``early_exit_ratio`` of final confidence
        * Entropy within ``early_exit_tolerance`` of final entropy
        * Margin within ``early_exit_tolerance`` of final margin

        Returns a dict with the layer index, the achieved
        confidence, and the targets. Returns ``None`` when no
        layer qualifies.
        """
        if not layers:
            return None
        final = layers[-1]
        target_confidence = self.early_exit_ratio * final.confidence
        target_entropy = final.entropy
        target_margin = final.margin
        # Tolerance band on entropy and margin. Asymmetric so a
        # noisy late layer doesn't disqualify a candidate whose
        # entropy is *almost* the final.
        entropy_tol = self.early_exit_tolerance * max(abs(target_entropy), 1e-9)
        margin_tol = self.early_exit_tolerance * max(abs(target_margin), 1e-9)

        for layer in layers:
            if layer.layer < self.early_exit_skip:
                continue
            if layer.confidence < target_confidence:
                continue
            if abs(layer.entropy - target_entropy) > entropy_tol:
                continue
            if abs(layer.margin - target_margin) > margin_tol:
                continue
            # Top1 must be stable from this layer to the end.
            tail = [l for l in layers if l.layer >= layer.layer]
            if any(l.top1_id != layer.top1_id for l in tail):
                continue
            return {
                "layer": layer.layer,
                "achieved_confidence": layer.confidence,
                "target_confidence": target_confidence,
                "entropy": layer.entropy,
                "margin": layer.margin,
                "final_confidence": final.confidence,
                "final_entropy": final.entropy,
                "final_margin": final.margin,
            }
        return None

    def compute(self) -> dict[str, Any]:
        """Return the analysis as a JSON-serialisable dict.

        The schema version is bumped to ``2.0`` to flag the
        prediction-evolution redesign; the previous hidden-state
        based LayerAnalyzer used ``1.0``. Downstream tools that
        add the schema version to their parsing should treat
        anything < 2.0 as the legacy format.
        """
        layers = self._ordered()
        if not layers:
            return {
                "schema_version": "2.0",
                "layer_count": 0,
                "layers": [],
            }

        final = layers[-1]
        stabilization = self._stabilization_layer(layers)
        most_influential = self._most_influential(layers)
        changes = self._prediction_changes(layers)
        early_exit = self._suggested_early_exit(layers)

        return {
            "schema_version": "2.0",
            "layer_count": len(layers),
            "final": {
                "layer": final.layer,
                "top1_id": final.top1_id,
                "top1": final.top1_text,
                "confidence": final.confidence,
                "entropy": final.entropy,
                "margin": final.margin,
            },
            "stabilization_layer": stabilization,
            "most_influential": [
                {
                    "layer": l.layer,
                    "delta_confidence": l.delta_confidence,
                    "top1": l.top1_text,
                }
                for l in most_influential
            ],
            "prediction_changes": [
                {
                    "layer": l.layer,
                    "from": l.prev_top1_text,
                    "to": l.top1_text,
                }
                for l in changes
            ],
            "total_changes": len(changes),
            "suggested_early_exit": early_exit,
            "confidence_growth": [
                {"layer": l.layer, "confidence": l.confidence}
                for l in layers
            ],
            "layers": [
                {
                    "layer": l.layer,
                    "top1_id": l.top1_id,
                    "top1": l.top1_text,
                    "top5": [
                        {"id": tid, "text": txt, "prob": prob}
                        for tid, txt, prob in zip(
                            l.top5_ids, l.top5_texts, l.top5_probs
                        )
                    ],
                    "confidence": l.confidence,
                    "entropy": l.entropy,
                    "margin": l.margin,
                    "prediction_changed": l.prediction_changed,
                    "prev_top1": l.prev_top1_text,
                    "delta_confidence": l.delta_confidence,
                }
                for l in layers
            ],
        }

    def summarize(self, stream: Any = None) -> dict[str, Any]:
        """Print the prediction-evolution report and return the dict.

        Output (stderr)::

            === Prediction Evolution ===
            L00  Top1=I        P=0.07  Entropy=12.50  Margin=0.02
            L01  Top1=I        P=0.11  ΔP=+0.04 ...
            L02  Top1=Okay     P=0.19  Top token changed ★ ...

            ================================================
            1. Prediction Stabilization Layer
            Prediction stabilizes at Layer 18

            ================================================
            2. Confidence Growth
            L00 0.07
            ...

            ================================================
            3. Most Influential Layers
            ΔConfidence
            L03 +0.17
            L11 +0.13
            L18 +0.09
            Ranking:
            1 L03
            2 L11
            3 L18

            ================================================
            4. Prediction Changes
            L02  I -> Okay
            L08  Okay -> Sure
            L12  Sure -> Certainly
            Total changes: 3

            ================================================
            5. Suggested Early Exit
            Final confidence: 0.91
            95% target: 0.8645
            First layer meeting all criteria: Layer 21
        """
        if stream is None:
            stream = sys.stderr
        report = self.compute()

        if report["layer_count"] == 0:
            stream.write("\n=== Prediction Evolution ===\n")
            stream.write("No layers recorded; pass --debug to populate.\n")
            return report

        sep = "=" * 60

        # 1. Per-layer table.
        stream.write("\n=== Prediction Evolution ===\n")
        for layer in self._layers:
            extra = ""
            if layer.prediction_changed:
                extra = f"  Top token changed ★  ({layer.prev_top1_text} -> {layer.top1_text})"
            if layer.delta_confidence != 0.0:
                delta_str = (
                    f"+{layer.delta_confidence:.3f}"
                    if layer.delta_confidence > 0
                    else f"{layer.delta_confidence:.3f}"
                )
                extra += f"  ΔP={delta_str}"
            stream.write(
                f"L{layer.layer:02d}  "
                f"Top1={layer.top1_text:<14} "
                f"P={layer.top1_prob:.3f}  "
                f"Entropy={layer.entropy:.3f}  "
                f"Margin={layer.margin:.3f}"
                f"{extra}\n"
            )

        # 2. Stabilization layer.
        stream.write(f"\n{sep}\n")
        stream.write("1. Prediction Stabilization Layer\n")
        if report["stabilization_layer"] is not None:
            stream.write(
                f"Prediction stabilizes at Layer {report['stabilization_layer']}\n"
            )
        else:
            stream.write("Top1 token kept changing; no stabilization point found.\n")

        # 3. Confidence growth.
        stream.write(f"\n{sep}\n")
        stream.write("2. Confidence Growth\n")
        for entry in report["confidence_growth"]:
            stream.write(f"L{entry['layer']:02d} {entry['confidence']:.3f}\n")

        # 4. Most Influential Layers.
        stream.write(f"\n{sep}\n")
        stream.write("3. Most Influential Layers\n")
        stream.write("ΔConfidence\n")
        for entry in report["most_influential"]:
            stream.write(
                f"  L{entry['layer']:02d} {entry['delta_confidence']:+.3f}  "
                f"(top1={entry['top1']})\n"
            )
        if report["most_influential"]:
            stream.write("Ranking:\n")
            for rank, entry in enumerate(report["most_influential"], 1):
                stream.write(f"  {rank} L{entry['layer']:02d}\n")

        # 5. Prediction Changes.
        stream.write(f"\n{sep}\n")
        stream.write("4. Prediction Changes\n")
        if report["prediction_changes"]:
            for change in report["prediction_changes"]:
                stream.write(
                    f"  L{change['layer']:02d}  "
                    f"{change['from']} -> {change['to']}\n"
                )
        else:
            stream.write("  No top-1 changes across layers.\n")
        stream.write(f"Total changes: {report['total_changes']}\n")

        # 6. Suggested Early Exit.
        stream.write(f"\n{sep}\n")
        stream.write("5. Suggested Early Exit\n")
        final = report["final"]
        target = (
            report["suggested_early_exit"]["target_confidence"]
            if report["suggested_early_exit"] is not None
            else self.early_exit_ratio * final["confidence"]
        )
        stream.write(f"Final confidence: {final['confidence']:.3f}\n")
        stream.write(
            f"{(self.early_exit_ratio * 100):.0f}% target: {target:.3f}\n"
        )
        if report["suggested_early_exit"] is not None:
            exit_info = report["suggested_early_exit"]
            stream.write(
                f"First layer meeting all criteria: Layer {exit_info['layer']}\n"
            )
            stream.write(
                f"Achieved confidence: {exit_info['achieved_confidence']:.3f}\n"
            )
            stream.write(
                f"  Entropy: {exit_info['entropy']:.3f} "
                f"(final {exit_info['final_entropy']:.3f})\n"
            )
            stream.write(
                f"  Margin:  {exit_info['margin']:.3f} "
                f"(final {exit_info['final_margin']:.3f})\n"
            )
        else:
            stream.write(
                "No layer meets all four criteria (top1 stable, "
                f"confidence >= {self.early_exit_ratio * 100:.0f}% of final, "
                "entropy stable, margin stable).\n"
            )

        stream.flush()
        return report

    def save(self, path: str | Path) -> Path:
        """Persist the analysis as JSON. Returns the resolved path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=2))
        return target

    def to_json(self, indent: int | None = 2) -> str:
        """Return the report as a JSON string."""
        return json.dumps(self.compute(), indent=indent)
