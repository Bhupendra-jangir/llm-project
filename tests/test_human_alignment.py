from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.human_alignment import (
    compute_inter_rater_agreement,
    compute_metric_alignment,
    create_stratified_subset,
)
from src.utils import load_summeval, records_to_dataframe


def test_human_subset_is_stratified_by_system_and_score_quartile():
    expert_scores = records_to_dataframe(load_summeval())
    subset = create_stratified_subset(expert_scores, n_samples=160, seed=42)
    ranked = expert_scores["human_overall"].rank(method="first")
    quartiles = pd.qcut(ranked, q=4, labels=False) + 1
    expected_strata = expert_scores[["doc_id", "system_id"]].copy()
    expected_strata["score_quartile"] = quartiles.to_numpy()
    selected_strata = subset[["doc_id", "system_id"]].merge(
        expected_strata,
        on=["doc_id", "system_id"],
        validate="one_to_one",
    )

    assert len(subset) == 160
    assert subset[["doc_id", "system_id"]].drop_duplicates().shape[0] == 160
    expected_groups = expected_strata.groupby(["system_id", "score_quartile"]).size()
    selected_groups = selected_strata.groupby(["system_id", "score_quartile"]).size()
    assert len(expected_groups) == 64
    assert len(selected_groups) == 64
    assert selected_groups.min() >= 1


def test_missing_new_ratings_are_not_simulated(tmp_path):
    expert_scores = records_to_dataframe(load_summeval(limit=4))
    result = compute_inter_rater_agreement(
        expert_scores,
        human_ratings_path=Path(tmp_path / "not_provided.csv"),
    )
    assert result is None


def test_automatic_metrics_align_with_expert_scores_without_duplicate_columns():
    expert_scores = records_to_dataframe(load_summeval(limit=16))
    automatic_metrics = expert_scores[["doc_id", "system_id", "human_overall"]].copy()
    automatic_metrics["rouge1_ref"] = range(len(automatic_metrics))

    alignment = compute_metric_alignment(automatic_metrics, expert_scores)

    assert set(alignment["metric"]) == {"rouge1_ref"}
    assert set(alignment["human_dim"]) == {
        "coherence",
        "consistency",
        "fluency",
        "relevance",
        "overall",
    }
