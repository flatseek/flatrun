"""Microsecond profiler for the forward pass.

Two usage modes:

* **Per-section timing** — wrap any code block in
  ``with profiler.section("rms_norm"): ...`` and the elapsed wall
  time accumulates in the named bucket. Sections are nested-friendly:
  nesting the same name inside itself adds the inner elapsed time to
  the outer bucket.

* **Per-layer accumulation** — call ``begin_layer(idx)`` before
  processing a layer and ``end_layer()`` after. The current
  layer's section times are copied into the per-layer log and
  the bucket is reset.

The profiler is opt-in: the forwarder only times sections when
``profiler=`` is passed. The default forwarder has zero overhead
from the profiler.

The output is human-readable text (stderr) plus a JSON dump for
post-processing. The JSON keys are stable across runs so a
notebook can correlate layers across many prompts.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class Profiler:
    """Microsecond-precision profiler for forward pass operations."""

    def __init__(self) -> None:
        # Per-layer log: list of (layer_idx, {section: total_us}) so
        # the user can see the layer-by-layer breakdown.
        self._per_layer: list[tuple[int, dict[str, float]]] = []
        self._current_layer_idx: int = -1
        self._current_secs: dict[str, float] = {}
        # Counter for nested ``section`` calls. Real nesting is
        # rare in the forward pass but the counter keeps the
        # bookkeeping honest.
        self._section_stack: list[tuple[str, float]] = []
        # Section names we've seen, in registration order, so the
        # summary table is stable.
        self._section_names: list[str] = []
        self._section_totals: dict[str, float] = {}
        self._frozen: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_layer(self, layer_idx: int) -> None:
        """Start a new layer's bucket. The previous layer's buckets
        are pushed into the per-layer log."""
        if self._current_secs:
            self._per_layer.append((self._current_layer_idx, self._current_secs))
        self._current_layer_idx = int(layer_idx)
        self._current_secs = {}

    def end_layer(self) -> None:
        """Close the current layer. The bucket is pushed into the
        per-layer log."""
        if self._current_secs:
            self._per_layer.append((self._current_layer_idx, self._current_secs))
        self._current_secs = {}
        self._current_layer_idx = -1

    def freeze(self) -> None:
        """Stop accumulating. The decoder blocks check ``_frozen``
        so the autoregressive decode loop (which re-enters the
        forwarder) becomes a no-op for the profiler just like the
        ``PredictionAnalyzer`` does."""
        if self._current_secs:
            self._per_layer.append((self._current_layer_idx, self._current_secs))
        self._current_secs = {}
        self._frozen = True

    # ------------------------------------------------------------------
    # Section timing
    # ------------------------------------------------------------------

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Time a block of code under ``name``.

        ``name`` is a free-form string. The elapsed wall-clock time
        in microseconds is added to the current layer's bucket. If
        ``name`` is new, it's seeded with 0.
        """
        if self._frozen:
            yield
            return
        self._section_stack.append((name, time.perf_counter()))
        try:
            yield
        finally:
            started = self._section_stack.pop()
            elapsed_us = (time.perf_counter() - started[1]) * 1_000_000.0
            self._record(started[0], elapsed_us)

    def _record(self, name: str, elapsed_us: float) -> None:
        if name not in self._section_totals:
            self._section_names.append(name)
            self._section_totals[name] = 0.0
        self._section_totals[name] += elapsed_us
        self._current_secs[name] = self._current_secs.get(name, 0.0) + elapsed_us

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def layer_count(self) -> int:
        return len(self._per_layer)

    def per_layer_sections(self) -> list[tuple[int, dict[str, float]]]:
        """Return a copy of the per-layer log. Mutating it does not
        affect the profiler."""
        return list(self._per_layer)

    def section_totals(self) -> dict[str, float]:
        """Return total time per section across all layers, in µs."""
        return dict(self._section_totals)

    def print_per_layer(self, stream: Any = None) -> None:
        """Print each layer's section breakdown to ``stream``."""
        if stream is None:
            stream = sys.stderr
        if not self._per_layer:
            stream.write("No layers recorded.\n")
            return
        for layer_idx, secs in self._per_layer:
            total = sum(secs.values())
            stream.write(f"\nLayer {layer_idx}\n")
            stream.write("-" * 33 + "\n")
            for name in self._section_names:
                us = secs.get(name, 0.0)
                if us <= 0.0:
                    continue
                stream.write(f"{name:<22} {us / 1000.0:8.3f} ms\n")
            stream.write(f"{'Layer Total':<22} {total / 1000.0:8.3f} ms\n")
        stream.flush()

    def print_summary(self, stream: Any = None) -> None:
        """Print the aggregated summary in the format the user asked for.

        Sections are aggregated into high-level categories (RMSNorm,
        Attention, MLP, ...) so the percentages are easier to
        interpret than a flat per-section list.
        """
        if stream is None:
            stream = sys.stderr
        totals = self.section_totals()
        agg = self._aggregate_to_categories(totals)
        total_us = sum(agg.values())
        if total_us <= 0.0:
            stream.write("No profiler data collected.\n")
            return

        stream.write("\n=========================\n")
        stream.write("PROFILE SUMMARY\n")
        stream.write("=========================\n")
        # Categories are sorted by their fraction descending so the
        # biggest contributor is at the top.
        for name, us in sorted(agg.items(), key=lambda kv: -kv[1]):
            pct = (us / total_us) * 100.0
            stream.write(f"  {name:<22} {pct:5.1f} %\n")

        # Top 20 slowest operations across all layers.
        top = sorted(totals.items(), key=lambda kv: -kv[1])[:20]
        stream.write("\nTop 20 Slowest Operations\n")
        for rank, (name, us) in enumerate(top, 1):
            stream.write(f"  {rank:>2}. {name}: {us / 1000.0:.2f} ms total\n")
        stream.flush()

    def _aggregate_to_categories(self, totals: dict[str, float]) -> dict[str, float]:
        """Roll the per-section times into high-level categories.

        The forwarder names sections after the operation they
        bracket — ``q_proj``, ``k_proj``, ``v_proj`` are Q/K/V
        projections; ``qkv_proj`` is the alternative fused path.
        The categories here group operations by their role in
        the layer so the percentages tell a story rather than
        enumerate every micro-section.
        """
        category_buckets: dict[str, list[str]] = {
            "Tensor Loading": ["load_tensors"],
            "Dequantization": ["dequant_q", "dequant_k", "dequant_v",
                              "dequant_o", "dequant_gate", "dequant_up",
                              "dequant_down", "dequant_attn_norm",
                              "dequant_mlp_norm", "dequant_q_norm",
                              "dequant_k_norm", "dequant_lm_head"],
            "Norm": ["rms_norm_attn", "rms_norm_mlp", "q_norm", "k_norm"],
            "Attention": ["qkv_proj", "q_proj", "k_proj", "v_proj",
                          "rope", "qk_matmul", "causal_mask",
                          "softmax", "av_matmul", "gqa_repeat",
                          "o_proj", "kv_append", "kv_stack"],
            "MLP": ["gateup_proj", "gate_proj", "up_proj",
                   "silu", "mul", "down_proj"],
            "Residual": ["residual_attn", "residual_mlp"],
            "Sampling": ["sample"],
            "Other": [],  # catch-all below
        }
        out: dict[str, float] = {cat: 0.0 for cat in category_buckets}
        for name, us in totals.items():
            placed = False
            for cat, members in category_buckets.items():
                if name in members:
                    out[cat] += us
                    placed = True
                    break
            if not placed:
                out["Other"] += us
        # Drop categories that were never hit (0.0) so the summary
        # is concise.
        return {k: v for k, v in out.items() if v > 0.0}

    def to_json(self) -> str:
        """Return the profiler state as a JSON string."""
        return json.dumps(self.compute(), indent=2)

    def compute(self) -> dict[str, Any]:
        """Return the profiler state as a JSON-serialisable dict."""
        per_layer = []
        for layer_idx, secs in self._per_layer:
            per_layer.append({
                "layer": layer_idx,
                "sections": {
                    name: secs.get(name, 0.0)
                    for name in self._section_names
                },
                "total_us": sum(secs.values()),
            })
        return {
            "sections": self._section_names,
            "totals_us": self._section_totals,
            "categories_us": self._aggregate_to_categories(self._section_totals),
            "per_layer": per_layer,
        }

    def save(self, path: str | Path) -> Path:
        """Persist the profiler state as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json())
        return target


__all__ = ["Profiler"]
