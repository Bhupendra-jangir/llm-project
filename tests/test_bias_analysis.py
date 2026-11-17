import pandas as pd

from src.bias_analysis import (
    compute_position_effect,
    compute_self_preference_bias,
    compute_system_score_distribution,
)


def test_system_score_distribution_reports_residual_effect_size():
    frame = pd.DataFrame(
        {
            "system_id": ["M0"] * 4 + ["M8"] * 4,
            "metric": [1, 2, 3, 4, 2, 3, 4, 5],
            "human": [1, 2, 3, 4, 1, 2, 3, 4],
        }
    )

    result = compute_system_score_distribution(frame, "metric", "human")

    assert len(result["per_system"]) == 2
    assert result["residual_anova_p_value"] >= 0
    assert result["residual_eta_squared"] >= 0


def test_unavailable_bias_analyses_are_reported_explicitly():
    frame = pd.DataFrame(
        {
            "doc_id": ["d1"],
            "system_id": ["M0"],
            "human_coherence": [3.0],
        }
    )

    self_preference = compute_self_preference_bias(
        judge_scores=None,
        human_df=frame,
        system_families={},
        judge_families={},
        human_dim="coherence",
    )

    assert self_preference["available"] is False
    assert compute_position_effect()["available"] is False
