"""
tests/test_unified_score.py - Unified score unit tests with leakage check.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.unified_score import compute_unified_score, DEFAULT_WEIGHTS, extract_components


class TestUnifiedScore:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "doc_id": [f"doc{i//10}" for i in range(n)],
            "system_id": [f"M{i%10}" for i in range(n)],
            "geval_overall": np.random.uniform(1, 5, n),
            "bertscore_F1_ref": np.random.uniform(0.6, 0.95, n),
            "rouge1_ref": np.random.uniform(0.1, 0.6, n),
            "human_overall": np.random.uniform(1, 5, n),
        })

    def test_range(self, sample_df):
        scores = compute_unified_score(sample_df, weights=DEFAULT_WEIGHTS)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_weights_sum_to_one(self):
        assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-6

    def test_no_human_input_leakage(self, sample_df):
        """Human scores must NOT be inputs to the unified score."""
        scores1 = compute_unified_score(sample_df)
        df2 = sample_df.copy()
        df2["human_overall"] = df2["human_overall"] * 2  # corrupt human labels
        scores2 = compute_unified_score(df2)
        # Scores must be IDENTICAL — human not used
        np.testing.assert_array_almost_equal(scores1, scores2, decimal=6)

    def test_components_normalized(self, sample_df):
        g, b, r, a = extract_components(sample_df)
        assert len(g) == len(sample_df)
        assert g.min() >= 0.0 and g.max() <= 1.0
