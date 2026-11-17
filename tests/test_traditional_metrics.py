"""
tests/test_traditional_metrics.py - Unit tests for traditional metrics.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.traditional_metrics import (
    compute_rouge,
    compute_bleu,
    compute_all_traditional_metrics,
    compute_bertscore_batch,
)
from src.utils import load_summeval


class TestROUGE:
    def test_perfect_match(self):
        result = compute_rouge("hello world", "hello world")
        assert result["rouge1"] > 0.99

    def test_no_overlap(self):
        result = compute_rouge("abc def", "xyz uvw")
        assert result["rouge1"] == 0.0

    def test_partial_overlap(self):
        result = compute_rouge("the cat sat on the mat", "the cat was on the mat")
        assert 0.5 < result["rouge1"] < 1.0

    def test_empty(self):
        result = compute_rouge("", "some reference text")
        assert result["rouge1"] == 0.0


class TestBLEU:
    def test_perfect(self):
        score = compute_bleu("hello world", "hello world")
        assert score > 0.9

    def test_zero(self):
        score = compute_bleu("abc def", "xyz uvw")
        assert score == 0.0

    def test_range(self):
        score = compute_bleu("the quick brown fox", "the quick brown dog")
        assert 0.0 <= score <= 1.0


def test_bertscore_requires_paired_inputs():
    with pytest.raises(ValueError, match="same length"):
        compute_bertscore_batch(["summary"], [])


def test_metric_cache_respects_requested_records(tmp_path, monkeypatch):
    import src.traditional_metrics as metrics

    calls = []

    def fake_bertscore(hypotheses, references, **kwargs):
        calls.append(len(hypotheses))
        values = [0.5] * len(hypotheses)
        return values, values, values

    monkeypatch.setattr(metrics, "_get_bertscore_model", lambda *_args: "test-model")
    monkeypatch.setattr(metrics, "compute_bertscore_batch", fake_bertscore)

    records = load_summeval(limit=3)
    cache_path = tmp_path / "traditional_metrics.csv"
    full = compute_all_traditional_metrics(
        records,
        output_path=cache_path,
        results_dir=tmp_path,
    )
    subset = compute_all_traditional_metrics(
        records,
        limit=2,
        output_path=cache_path,
        results_dir=tmp_path,
    )

    assert len(full) == 3
    assert len(subset) == 2
    assert calls == [3, 3]
    assert subset["doc_id"].tolist() == [record["doc_id"] for record in records[:2]]
