"""Tests for the per-token debug helpers and prediction analyzer."""

from __future__ import annotations

import io
import json

import numpy as np

from flatrun.utils.debug import (
    PredictionAnalyzer,
    per_token_metrics,
    print_per_token_table,
)


def _fake_tokenizer():
    """Tiny dictionary tokenizer for tests."""

    class _Tok:
        def decode(self, ids):
            return {0: "<bos>", 1: "Hello", 2: "world", 3: "!"}.get(int(ids[0]), f"tok{int(ids[0])}")

    return _Tok()


def test_per_token_metrics_shape_and_basic_columns() -> None:
    h = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    prev_h = np.zeros_like(h)
    tokens = np.array([0, 1, 2], dtype=np.int64)

    metrics = per_token_metrics(h, prev_h, tokens=tokens, logits=None)

    np.testing.assert_allclose(metrics["norms"], np.ones(3))
    np.testing.assert_allclose(metrics["deltas"], np.ones(3))
    np.testing.assert_allclose(metrics["stables"], np.zeros(3))
    assert metrics["entropies"] is None
    assert metrics["confs"] is None
    np.testing.assert_array_equal(metrics["tokens"], tokens)


def test_per_token_metrics_with_logits() -> None:
    h = np.eye(3, dtype=np.float32)
    prev_h = None
    logits = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=np.float32,
    )
    metrics = per_token_metrics(h, prev_h, tokens=None, logits=logits)
    np.testing.assert_allclose(metrics["entropies"], np.zeros(3), atol=1e-3)
    np.testing.assert_allclose(metrics["confs"], np.ones(3), atol=1e-3)


def test_per_token_metrics_stable_cosine() -> None:
    h = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float32)
    prev_h = np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    metrics = per_token_metrics(h, prev_h, tokens=None, logits=None)
    expected_0 = 1.0 / (np.sqrt(1.25) * 1.0)
    expected_1 = 0.5 / (1.0 * 0.5)
    assert abs(metrics["stables"][0] - expected_0) < 1e-5
    assert abs(metrics["stables"][1] - expected_1) < 1e-5


def test_per_token_metrics_nan_deltas_when_no_prev() -> None:
    h = np.ones((2, 4), dtype=np.float32)
    metrics = per_token_metrics(h, None, tokens=None, logits=None)
    assert all(np.isnan(d) for d in metrics["deltas"])
    assert all(np.isnan(s) for s in metrics["stables"])


def test_print_per_token_table_layout() -> None:
    h = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    prev_h = np.zeros_like(h)
    tokens = np.array([0, 1, 2], dtype=np.int64)
    metrics = per_token_metrics(h, prev_h, tokens=tokens, logits=None)
    buf = io.StringIO()
    print_per_token_table(
        5, metrics, tokenizer=_fake_tokenizer(), max_rows=8, stream=buf,
    )
    out = buf.getvalue()
    assert "[dbg L 5] per-token (sorted by norm DESC, showing 3/3):" in out
    assert "rank" in out
    assert "decode" in out
    assert out.count("3.000") >= 1
    assert "Hello" in out
    assert "<bos>" in out
    first_row = [line for line in out.splitlines() if line.strip().startswith("1")][0]
    assert "3.000" in first_row


def test_print_per_token_table_placeholder_columns() -> None:
    h = np.ones((1, 4), dtype=np.float32)
    metrics = per_token_metrics(h, None, tokens=np.array([0]), logits=None)
    buf = io.StringIO()
    print_per_token_table(0, metrics, tokenizer=_fake_tokenizer(), stream=buf)
    out = buf.getvalue()
    row = [line for line in out.splitlines() if line.strip().startswith("1")][0]
    assert row.count("-") >= 2


def test_print_per_token_table_with_logits() -> None:
    h = np.eye(2, dtype=np.float32)
    logits = np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    metrics = per_token_metrics(h, None, tokens=np.array([0, 1]), logits=logits)
    buf = io.StringIO()
    print_per_token_table(0, metrics, tokenizer=_fake_tokenizer(), stream=buf)
    out = buf.getvalue()
    row = [line for line in out.splitlines() if line.strip().startswith("1")][0]
    fields = row.split("  ")
    assert "-" not in fields[-4]
    assert "-" not in fields[-3]


def test_print_per_token_table_respects_max_rows() -> None:
    h = np.arange(20, dtype=np.float32).reshape(20, 1)
    tokens = np.arange(20, dtype=np.int64)
    metrics = per_token_metrics(h, None, tokens=tokens, logits=None)
    buf = io.StringIO()
    print_per_token_table(
        0, metrics, tokenizer=_fake_tokenizer(), max_rows=5, stream=buf,
    )
    out = buf.getvalue()
    assert "showing 5/20" in out
    body_lines = [l for l in out.splitlines() if l.strip().startswith(("1", "2", "3", "4", "5"))]
    assert len(body_lines) == 5


def test_print_per_token_table_excludes_special_tokens() -> None:
    h = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    tokens = np.array([0, 1, 2], dtype=np.int64)
    metrics = per_token_metrics(h, None, tokens=tokens, logits=None)
    buf = io.StringIO()
    print_per_token_table(
        0,
        metrics,
        tokenizer=_fake_tokenizer(),
        exclude_ids={0},
        stream=buf,
    )
    out = buf.getvalue()
    assert "excluded 1 special" in out
    header_line = [l for l in out.splitlines() if "showing" in l][0]
    assert "showing 2/2" in header_line
    body_lines = [
        l for l in out.splitlines()
        if l.strip().startswith(("1", "2", "3")) and "rank" not in l
    ]
    assert len(body_lines) == 2
    for line in body_lines:
        assert "<bos>" not in line


def test_print_per_token_table_no_exclude_no_change() -> None:
    h = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    tokens = np.array([0, 1], dtype=np.int64)
    metrics = per_token_metrics(h, None, tokens=tokens, logits=None)
    buf = io.StringIO()
    print_per_token_table(0, metrics, tokenizer=_fake_tokenizer(), stream=buf)
    out = buf.getvalue()
    assert "special" not in out
    assert "showing 2/2" in out


# ---------------------------------------------------------------------------
# PredictionAnalyzer
# ---------------------------------------------------------------------------


def _predictions_to_logit(top1_id: int, vocab_size: int = 16) -> np.ndarray:
    """Return a (seq_len, vocab) logit tensor with a single peak.

    Helper for building deterministic per-layer predictions. The
    softmax is sharp enough that argmax is unambiguous.
    """
    logits = np.full((1, vocab_size), -10.0, dtype=np.float32)
    logits[0, top1_id] = 10.0
    return logits


def _record_promotion(analyzer: PredictionAnalyzer, layer: int, top1_id: int) -> None:
    """Feed a deterministic prediction for one layer."""
    analyzer.record(layer, _predictions_to_logit(top1_id), tokenizer=_fake_tokenizer())


def _make_promotion_analyzer(layer_count: int = 8) -> PredictionAnalyzer:
    """Build an analyzer where each layer "promotes" to a different top1.

    Layer 0: top1=0 (P=0.999)
    Layer 1: top1=1 (P=0.999)
    ...
    A monotonic ``I -> Okay -> Sure -> Certainly`` style progression.
    """
    a = PredictionAnalyzer(layer_count=layer_count)
    for i in range(layer_count):
        _record_promotion(a, i, top1_id=i % 4)
    return a


def _make_stable_analyzer(layer_count: int = 8) -> PredictionAnalyzer:
    """Build an analyzer where top1 stabilises early.

    Top1=0 from layer 0 onwards. Confidence grows: 0.5, 0.6, 0.7, ...
    """
    a = PredictionAnalyzer(layer_count=layer_count)
    for i in range(layer_count):
        # Higher confidence each layer. We build a custom logits
        # tensor so the resulting softmax probability is what we
        # want.
        logit_value = float(i + 1)  # 1, 2, 3, ...
        logits = np.full((1, 16), 0.0, dtype=np.float32)
        logits[0, 0] = logit_value
        # All other logits are 0, so softmax(top1) = exp(logit_value) /
        # (exp(logit_value) + 15).
        a.record(i, logits, tokenizer=_fake_tokenizer())
    return a


def test_prediction_analyzer_records_each_layer_once() -> None:
    a = _make_promotion_analyzer(8)
    assert a.layer_count() == 8
    # Re-recording the same layer is a no-op. We use top1_id=1
    # (matches the fake tokenizer's vocabulary).
    _record_promotion(a, 0, top1_id=1)
    assert a.layer_count() == 8


def test_prediction_analyzer_freeze_stops_recording() -> None:
    a = _make_promotion_analyzer(8)
    a.freeze()
    _record_promotion(a, 99, top1_id=0)
    assert a.layer_count() == 8


def test_prediction_analyzer_predicts_top1() -> None:
    a = PredictionAnalyzer(layer_count=3)
    # Layer 0: top1 = 0 (<bos>)
    a.record(0, _predictions_to_logit(0), tokenizer=_fake_tokenizer())
    # Layer 1: top1 = 1 (Hello)
    a.record(1, _predictions_to_logit(1), tokenizer=_fake_tokenizer())
    layers = a._ordered()
    assert layers[0].top1_id == 0
    assert layers[0].top1_text == "<bos>"
    assert layers[1].top1_id == 1
    assert layers[1].top1_text == "Hello"


def test_prediction_analyzer_prediction_changed_flag() -> None:
    a = PredictionAnalyzer(layer_count=3)
    a.record(0, _predictions_to_logit(0), tokenizer=_fake_tokenizer())
    a.record(1, _predictions_to_logit(0), tokenizer=_fake_tokenizer())  # same
    a.record(2, _predictions_to_logit(1), tokenizer=_fake_tokenizer())  # changed
    layers = a._ordered()
    assert layers[0].prediction_changed is False
    assert layers[1].prediction_changed is False
    assert layers[2].prediction_changed is True
    assert layers[2].prev_top1_id == 0


def test_prediction_analyzer_delta_confidence() -> None:
    a = _make_stable_analyzer(5)
    layers = a._ordered()
    # Confidence grows monotonically (we built it that way), so every
    # delta_confidence from layer 1 onward is positive. Layer 0 has
    # delta_confidence = 0 because there is no previous layer.
    assert layers[0].delta_confidence == 0.0
    for layer in layers[1:]:
        assert layer.delta_confidence > 0.0


def test_prediction_analyzer_stabilization_layer() -> None:
    """First layer where top1 stops changing and confidence keeps increasing."""
    a = _make_stable_analyzer(8)
    report = a.compute()
    # Top1=0 throughout, so stabilization is layer 0 (nothing
    # changes from there on).
    assert report["stabilization_layer"] == 0


def test_prediction_analyzer_stabilization_layer_with_early_change() -> None:
    """Top1 changes once, then settles. Stabilization should be the
    layer *after* the change."""
    a = PredictionAnalyzer(layer_count=6)
    # Layer 0: top1=0
    a.record(0, _predictions_to_logit(0), tokenizer=_fake_tokenizer())
    # Layer 1: top1=1 (change)
    a.record(1, _predictions_to_logit(1), tokenizer=_fake_tokenizer())
    # Layers 2-5: top1=1 with growing confidence
    confidences = [0.5, 0.6, 0.7, 0.8]
    for i, conf in enumerate(confidences, start=2):
        # Build logits that give roughly the desired confidence.
        # log(p) = logit_value - log(15 * exp(0) + exp(logit_value))
        # We reverse-engineer using a simple two-value softmax.
        logits = np.full((1, 16), 0.0, dtype=np.float32)
        logits[0, 1] = float(conf) * 4.0
        a.record(i, logits, tokenizer=_fake_tokenizer())
    report = a.compute()
    # Stabilization: from layer 2 onwards, top1 is stable and
    # confidence keeps increasing.
    assert report["stabilization_layer"] == 2


def test_prediction_analyzer_most_influential() -> None:
    """Most influential layers should be ranked by delta_confidence."""
    a = _make_stable_analyzer(8)
    report = a.compute()
    # Every layer's delta_confidence is positive. The first layer
    # delta is 0 (no previous layer), so the top-k should skip it.
    # Just check the list isn't empty.
    assert len(report["most_influential"]) > 0
    # Sorted by delta_confidence descending
    deltas = [entry["delta_confidence"] for entry in report["most_influential"]]
    assert deltas == sorted(deltas, reverse=True)


def test_prediction_analyzer_prediction_changes() -> None:
    a = _make_promotion_analyzer(8)
    report = a.compute()
    # Top1 changes every layer (each layer promotes a different
    # token), so there should be 7 changes (layer 0 has no change).
    assert report["total_changes"] == 7
    # The first change is layer 1 (from <bos> to Hello)
    assert report["prediction_changes"][0]["layer"] == 1
    assert report["prediction_changes"][0]["from"] == "<bos>"
    assert report["prediction_changes"][0]["to"] == "Hello"


def test_prediction_analyzer_suggested_early_exit() -> None:
    """First layer where top1 is stable, confidence >= 95% of final,
    entropy and margin are stable."""
    a = _make_stable_analyzer(20)
    report = a.compute()
    # Final confidence is the highest (last layer). The first layer
    # where confidence >= 0.95 * final_confidence AND top1 is
    # stable from there on. With our stable pattern, the
    # accommodation is straightforward.
    assert report["suggested_early_exit"] is not None
    exit_info = report["suggested_early_exit"]
    assert "achieved_confidence" in exit_info
    assert "target_confidence" in exit_info


def test_prediction_analyzer_summarize_output() -> None:
    a = _make_promotion_analyzer(8)
    buf = io.StringIO()
    a.summarize(stream=buf)
    out = buf.getvalue()
    assert "Prediction Evolution" in out
    assert "Prediction Stabilization Layer" in out
    assert "Confidence Growth" in out
    assert "Most Influential Layers" in out
    assert "Prediction Changes" in out
    assert "Suggested Early Exit" in out


def test_prediction_analyzer_to_json_schema() -> None:
    a = _make_promotion_analyzer(5)
    raw = a.to_json()
    parsed = json.loads(raw)
    assert parsed["schema_version"] == "2.0"
    assert parsed["layer_count"] == 5
    # Per-layer entries have the documented fields
    layer = parsed["layers"][0]
    assert "top1" in layer
    assert "top5" in layer
    assert "confidence" in layer
    assert "entropy" in layer
    assert "margin" in layer
    assert "prediction_changed" in layer
    assert "delta_confidence" in layer
    # top5 is a list of {id, text, prob}
    for entry in layer["top5"]:
        assert "id" in entry
        assert "text" in entry
        assert "prob" in entry


def test_prediction_analyzer_save(tmp_path) -> None:
    a = _make_promotion_analyzer(5)
    out = a.save(tmp_path / "analysis.json")
    assert out.exists()
    parsed = json.loads(out.read_text())
    assert parsed["layer_count"] == 5
    assert parsed["schema_version"] == "2.0"


def test_prediction_analyzer_no_result_when_empty() -> None:
    a = PredictionAnalyzer(layer_count=0)
    report = a.compute()
    assert report["layer_count"] == 0
    assert report["layers"] == []
    buf = io.StringIO()
    summary = a.summarize(stream=buf)
    assert "No layers recorded" in buf.getvalue()
    assert summary["layer_count"] == 0
