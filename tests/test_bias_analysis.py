import numpy as np
import pandas as pd

from src.bias_analysis import (
    compute_length_bias,
    compute_position_effect,
    compute_self_preference_bias,
    compute_system_score_distribution,
)


def test_length_bias_supports_current_pingouin_p_value_column():
    frame = pd.DataFrame({
        "metric": [1, 3, 2, 6, 5, 7, 9, 8, 11, 10, 12, 4],
        "human": [5, 4, 6, 2, 3, 7, 8, 5, 9, 1, 6, 10],
        "system_output": ["word " * count for count in range(1, 13)],
    })

    result = compute_length_bias(frame, "metric", "human")

    assert all(np.isfinite(value) for value in result.values())


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
