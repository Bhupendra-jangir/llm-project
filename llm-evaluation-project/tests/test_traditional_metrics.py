"""
tests/test_traditional_metrics.py - Unit tests for traditional metrics.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.traditional_metrics import compute_rouge, compute_bleu


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
