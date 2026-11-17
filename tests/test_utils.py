"""
tests/test_utils.py - Unit tests for src/utils.py
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import (
    parse_score, summary_level_correlation, pooled_correlation,
    bootstrap_ci, z_score_normalize,
    minmax_normalize, DiskCache, load_summeval,
    validate_summeval, normalize_summeval_record, records_to_dataframe, CostTracker
)


class TestParseScore:
    def test_integer(self):
        assert parse_score("3") == 3.0

    def test_float(self):
        assert parse_score("3.5") == 3.5

    def test_leading_dash(self):
        assert parse_score("- 4") == 4.0

    def test_colon_prefix(self):
        assert parse_score("Coherence: 3") == 3.0

    def test_out_of_range(self):
        result = parse_score("7", min_score=1, max_score=5)
        assert result is None

    def test_empty_string(self):
        assert parse_score("") is None

    def test_text_only(self):
        assert parse_score("no numbers here") is None


class TestCorrelations:
    def test_known_correlation_values_match_scipy(self):
        from scipy import stats

        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        human = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        doc_ids = np.array(["doc"] * 5)

        summary = summary_level_correlation(pred, human, doc_ids)
        pooled = pooled_correlation(pred, human)

        assert summary["spearman"] == pytest.approx(stats.spearmanr(pred, human).statistic)
        assert summary["pearson"] == pytest.approx(stats.pearsonr(pred, human).statistic)
        assert summary["kendall"] == pytest.approx(stats.kendalltau(pred, human).statistic)
        assert pooled["spearman"] == pytest.approx(stats.spearmanr(pred, human).statistic)
        assert pooled["pearson"] == pytest.approx(stats.pearsonr(pred, human).statistic)
        assert pooled["kendall"] == pytest.approx(stats.kendalltau(pred, human).statistic)

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        pred = np.random.randn(n)
        human = pred + np.random.randn(n) * 0.3
        doc_ids = np.array([f"doc{i//10}" for i in range(n)])
        system_ids = np.array([f"M{i%10}" for i in range(n)])
        return pred, human, doc_ids, system_ids

    def test_summary_level_spearman_high(self, sample_data):
        pred, human, doc_ids, system_ids = sample_data
        result = summary_level_correlation(pred, human, doc_ids)
        assert "spearman" in result
        assert result["spearman"] > 0.5

    def test_pooled_matches_scipy(self, sample_data):
        pred, human, doc_ids, system_ids = sample_data
        from scipy.stats import spearmanr
        result = pooled_correlation(pred, human)
        expected = spearmanr(pred, human).statistic
        assert abs(result["spearman"] - expected) < 1e-6

    def test_bootstrap_ci_coverage(self, sample_data):
        pred, human, doc_ids, system_ids = sample_data
        def _spearman(p, h, d):
            return summary_level_correlation(p, h, d)["spearman"]
        point, lo, hi = bootstrap_ci(pred, human, doc_ids, _spearman, n_resamples=50)
        assert lo <= point <= hi

    def test_bootstrap_rejects_misaligned_inputs(self):
        with pytest.raises(ValueError, match="equal lengths"):
            bootstrap_ci(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array(["doc", "doc"]),
                lambda p, h, d: 0.0,
                n_resamples=10,
            )

    def test_constant_pred_no_crash(self):
        pred = np.ones(50)
        human = np.random.randn(50)
        doc_ids = np.array([f"doc{i%5}" for i in range(50)])
        result = summary_level_correlation(pred, human, doc_ids)
        assert result["n_docs"] == 0 or np.isnan(result["spearman"])


class TestNormalization:
    def test_minmax(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        norm = minmax_normalize(arr)
        assert abs(norm.min()) < 1e-9
        assert abs(norm.max() - 1.0) < 1e-9

    def test_zscore(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        norm = z_score_normalize(arr)
        assert abs(norm.mean()) < 1e-9

    def test_constant_no_nan(self):
        arr = np.ones(5)
        assert not np.isnan(minmax_normalize(arr)).any()
        assert not np.isnan(z_score_normalize(arr)).any()


class TestDiskCache:
    def test_set_get(self, tmp_path):
        cache = DiskCache(tmp_path / "test.db")
        key = cache.make_key(prompt="test", model="gpt")
        cache.set(key, {"score": 3.5})
        result = cache.get(key)
        assert result == {"score": 3.5}

    def test_contains(self, tmp_path):
        cache = DiskCache(tmp_path / "test.db")
        key = cache.make_key(prompt="hello")
        assert key not in cache
        cache.set(key, 42)
        assert key in cache

    def test_idempotent_key(self):
        k1 = DiskCache.make_key(prompt="abc", model="gpt")
        k2 = DiskCache.make_key(prompt="abc", model="gpt")
        assert k1 == k2


class TestDataLoading:
    def test_load_summeval_count(self):
        data = load_summeval()
        assert len(data) == 1600
        assert len({record["doc_id"] for record in data}) == 100
        assert len({record["system_id"] for record in data}) == 16

    def test_validate_schema(self):
        data = load_summeval(limit=10)
        validate_summeval(data)

    def test_dataframe_shape(self):
        data = load_summeval(limit=10)
        df = records_to_dataframe(data)
        assert len(df) == 10
        assert "human_coherence" in df.columns

    def test_normalizes_alias_fields(self):
        record = {
            "document_id": 8,
            "model_id": "M0",
            "document": "source",
            "reference_summary": "reference",
            "summary": "candidate",
            "human_scores": {
                "coherence": 3,
                "consistency": 4,
                "fluency": 2,
                "relevance": 4,
                "overall": 3.25,
            },
        }

        normalized = normalize_summeval_record(record)

        assert normalized["doc_id"] == "8"
        assert normalized["system_id"] == "M0"
        assert normalized["system_output"] == "candidate"
        assert normalized["scores"]["overall"] == 3.25

    def test_rejects_duplicate_pairs_and_out_of_range_scores(self):
        data = load_summeval(limit=2)
        duplicate = [data[0], data[0]]
        with pytest.raises(ValueError, match="Duplicate"):
            validate_summeval(duplicate)

        invalid = [dict(data[0], scores={**data[0]["scores"], "fluency": 6})]
        with pytest.raises(ValueError, match="fluency score"):
            validate_summeval(invalid)


class TestCostTracker:
    def test_tracking(self):
        tracker = CostTracker("gpt-4o")
        tracker.log(1000, 100)
        assert tracker.input_tokens == 1000
        assert tracker.calls == 1
        assert tracker.estimated_cost_usd > 0
