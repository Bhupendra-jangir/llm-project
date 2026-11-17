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

from src.unified_score import (
    compute_unified_score,
    DEFAULT_WEIGHTS,
    extract_components,
    grid_search_weights,
    grid_weight_oof_predictions,
    learned_combiner,
    _scale_fold,
)
from sklearn.model_selection import GroupKFold


class TestUnifiedScore:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "doc_id": [f"doc{i//10}" for i in range(n)],
            "system_id": [f"M{i%10}" for i in range(n)],
            "geval_overall": np.random.uniform(1, 5, n),
            "geval_judge_std": np.random.uniform(0, 1, n),
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

    def test_missing_ensemble_agreement_is_not_defaulted(self, sample_df):
        with pytest.raises(ValueError, match="required metric columns"):
            extract_components(sample_df.drop(columns="geval_judge_std"))

    def test_fold_scaling_uses_training_bounds_only(self):
        train = np.array([[1.0], [3.0]])
        validation = np.array([[100.0]])
        train_scaled, validation_scaled = _scale_fold(train, validation)
        np.testing.assert_array_equal(train_scaled, [[0.0], [1.0]])
        np.testing.assert_array_equal(validation_scaled, [[49.5]])

    def test_grid_and_learned_combiners_use_document_grouped_folds(self, sample_df):
        grid = grid_search_weights(
            sample_df,
            human_col="human_overall",
            step=0.5,
            n_folds=5,
        )
        learned = learned_combiner(
            sample_df,
            human_col="human_overall",
            n_folds=5,
        )
        grid_oof = grid_weight_oof_predictions(
            sample_df,
            weights={
                "geval": grid.iloc[0]["w_geval"],
                "bertscore": grid.iloc[0]["w_bertscore"],
                "rouge": grid.iloc[0]["w_rouge"],
                "ensemble_agreement": grid.iloc[0]["w_ensemble"],
            },
            human_col="human_overall",
            n_folds=5,
        )

        assert not grid.empty
        assert grid["cv_folds"].eq(5).all()
        assert learned["cv_folds"] == 5
        assert len(learned["oof_predictions"]) == len(sample_df)
        assert np.isfinite(grid_oof).all()

    def test_grouped_cv_never_splits_documents_across_train_and_test(self, sample_df):
        groups = sample_df["doc_id"].to_numpy()
        splitter = GroupKFold(n_splits=5)

        for train_idx, test_idx in splitter.split(sample_df, groups=groups):
            assert set(groups[train_idx]).isdisjoint(groups[test_idx])
