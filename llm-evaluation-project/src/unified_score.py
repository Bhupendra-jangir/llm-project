"""
src/unified_score.py
====================
Unified evaluation score combining multiple metrics (Experiment G).

METHODOLOGICAL NOTE: Per DECISIONS.md, the original proposal's
"HumanAlignment" term is replaced with "LLM-judge ensemble agreement"
(inverse std of scores across multiple judge models), making the unified
score fully automatic without leaking any human labels.

Variants:
  (a) Fixed-weight combination: 0.40*G-Eval + 0.25*BERTScore + 0.15*ROUGE + 0.20*EnsembleAgreement
  (b) Grid search over the weight simplex (step=0.05)
  (c) Ridge regression + optional GradientBoosting with 5-fold GroupKFold CV

All components are min-max normalized before combination.
Comparison against individual metrics uses: sample-level, summary-level,
system-level correlations with bootstrap CIs.
"""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler

from src.utils import (
    DIMENSIONS,
    PROJECT_ROOT,
    ensure_results_dir,
    minmax_normalize,
    summary_level_correlation,
    pooled_correlation,
    system_level_correlation,
    bootstrap_ci,
    save_results,
)

logger = logging.getLogger(__name__)

# Default weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "geval": 0.40,
    "bertscore": 0.25,
    "rouge": 0.15,
    "ensemble_agreement": 0.20,
}


# ---------------------------------------------------------------------------
# Component extraction
# ---------------------------------------------------------------------------

def extract_components(
    df: pd.DataFrame,
    geval_col: str = "geval_overall",
    bertscore_col: str = "bertscore_F1_ref",
    rouge_col: str = "rouge1_ref",
    ensemble_std_col: Optional[str] = "geval_judge_std",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract and min-max normalize the four components.

    Args:
        df:                 DataFrame with all metric columns.
        geval_col:          G-Eval score column.
        bertscore_col:      BERTScore F1 column.
        rouge_col:          ROUGE score column.
        ensemble_std_col:   Std of judge ensemble scores (None → zeros).

    Returns:
        (geval_norm, bertscore_norm, rouge_norm, agreement_norm) each shape (N,).
    """
    def safe_norm(arr: np.ndarray) -> np.ndarray:
        arr = np.array(arr, dtype=float)
        nan_mask = np.isnan(arr)
        if nan_mask.all():
            return np.zeros_like(arr)
        # Fill NaN with column mean before normalizing
        arr[nan_mask] = np.nanmean(arr)
        return minmax_normalize(arr)

    geval = safe_norm(df[geval_col].values if geval_col in df.columns else np.full(len(df), 0.5))
    bertscore = safe_norm(df[bertscore_col].values if bertscore_col in df.columns else np.zeros(len(df)))
    rouge = safe_norm(df[rouge_col].values if rouge_col in df.columns else np.zeros(len(df)))

    if ensemble_std_col and ensemble_std_col in df.columns:
        # Invert std → agreement (high std = low agreement)
        std_vals = df[ensemble_std_col].values.astype(float)
        agreement = safe_norm(1.0 - std_vals)  # higher = more agreement
    else:
        # Fallback: use 1.0 for all (maximum agreement signal)
        agreement = np.ones(len(df))

    return geval, bertscore, rouge, agreement


# ---------------------------------------------------------------------------
# (a) Fixed-weight combination
# ---------------------------------------------------------------------------

def compute_unified_score(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Compute the weighted unified score.

    Args:
        df:      DataFrame with component columns.
        weights: Dict with keys geval, bertscore, rouge, ensemble_agreement.

    Returns:
        Unified score array, shape (N,).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    geval, bertscore, rouge, agreement = extract_components(df, **kwargs)
    score = (
        weights.get("geval", 0) * geval
        + weights.get("bertscore", 0) * bertscore
        + weights.get("rouge", 0) * rouge
        + weights.get("ensemble_agreement", 0) * agreement
    )
    return score


# ---------------------------------------------------------------------------
# (b) Grid search
# ---------------------------------------------------------------------------

def grid_search_weights(
    df: pd.DataFrame,
    human_col: str,
    doc_id_col: str = "doc_id",
    step: float = 0.05,
    **extract_kwargs: Any,
) -> pd.DataFrame:
    """Grid search over the weight simplex to maximise summary-level Spearman.

    Args:
        df:         DataFrame with all components and human_col.
        human_col:  Human ground truth column.
        step:       Grid step size (0.05 → 21^3 / 24 ≈ 1771 valid combos).

    Returns:
        DataFrame of all weight combinations sorted by Spearman descending.
    """
    geval, bertscore, rouge, agreement = extract_components(df, **extract_kwargs)
    human = df[human_col].values.astype(float)
    doc_ids = df[doc_id_col].values

    vals = np.arange(0, 1 + step, step)
    rows = []
    for w_g, w_b, w_r in product(vals, vals, vals):
        w_a = 1.0 - w_g - w_b - w_r
        if w_a < -1e-9 or w_a > 1 + 1e-9:
            continue
        w_a = max(0.0, min(1.0, w_a))

        score = w_g * geval + w_b * bertscore + w_r * rouge + w_a * agreement
        corr = summary_level_correlation(score, human, doc_ids)
        rows.append({
            "w_geval": round(w_g, 4),
            "w_bertscore": round(w_b, 4),
            "w_rouge": round(w_r, 4),
            "w_ensemble": round(w_a, 4),
            "spearman": corr["spearman"],
            "kendall": corr["kendall"],
            "pearson": corr["pearson"],
        })

    result = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    return result


# ---------------------------------------------------------------------------
# (c) Learned combiner
# ---------------------------------------------------------------------------

def learned_combiner(
    df: pd.DataFrame,
    human_col: str,
    doc_id_col: str = "doc_id",
    n_folds: int = 5,
    seed: int = 42,
    use_gradient_boosting: bool = False,
    **extract_kwargs: Any,
) -> Dict[str, Any]:
    """Train a linear/GB combiner with GroupKFold CV.

    Groups are doc_id to prevent data leakage.

    Returns:
        Dict with per-fold metrics, feature importances, and OOF predictions.
    """
    geval, bertscore, rouge, agreement = extract_components(df, **extract_kwargs)
    human = df[human_col].values.astype(float)
    doc_ids = df[doc_id_col].values

    # Remove NaN targets
    valid = ~np.isnan(human)
    X = np.column_stack([geval, bertscore, rouge, agreement])[valid]
    y = human[valid]
    groups = doc_ids[valid]

    gkf = GroupKFold(n_splits=n_folds)
    oof_preds = np.full(len(y), float("nan"))
    fold_metrics = []
    all_coefs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if use_gradient_boosting:
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(n_estimators=100, random_state=seed)
                model.fit(X_tr, y_tr)
                importances = model.feature_importances_
            except Exception as e:
                logger.warning(f"GB failed: {e}; falling back to Ridge")
                model = Ridge(alpha=1.0, random_state=seed)
                model.fit(X_tr, y_tr)
                importances = model.coef_
        else:
            model = Ridge(alpha=1.0, random_state=seed)
            model.fit(X_tr, y_tr)
            importances = model.coef_

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        all_coefs.append(importances)

        rho, _ = stats.spearmanr(preds, y_val)
        fold_metrics.append({"fold": fold, "spearman": float(rho)})

    mean_spearman = float(np.nanmean([m["spearman"] for m in fold_metrics]))
    mean_coefs = np.mean(all_coefs, axis=0)

    return {
        "model_type": "gradient_boosting" if use_gradient_boosting else "ridge",
        "cv_folds": n_folds,
        "fold_metrics": fold_metrics,
        "mean_cv_spearman": mean_spearman,
        "feature_names": ["geval", "bertscore", "rouge", "ensemble_agreement"],
        "mean_feature_importances": mean_coefs.tolist(),
        "oof_predictions": oof_preds.tolist(),
    }


# ---------------------------------------------------------------------------
# Full comparison
# ---------------------------------------------------------------------------

def compare_all_metrics(
    df: pd.DataFrame,
    human_col: str,
    metric_cols: List[str],
    unified_col: str,
    doc_id_col: str = "doc_id",
    system_id_col: str = "system_id",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare unified score vs all individual metrics.

    Returns DataFrame with one row per metric, columns:
    metric, sample_spearman, summary_spearman, system_spearman, is_better_than_{each}.
    """
    human = df[human_col].values.astype(float)
    doc_ids = df[doc_id_col].values
    system_ids = df[system_id_col].values

    all_metrics = metric_cols + [unified_col]
    rows = []

    for col in all_metrics:
        if col not in df.columns:
            continue
        pred = df[col].values.astype(float)
        valid = ~(np.isnan(pred) | np.isnan(human))
        p, h, d, s = pred[valid], human[valid], doc_ids[valid], system_ids[valid]

        sample_corr = pooled_correlation(p, h)
        summary_corr = summary_level_correlation(p, h, d)
        system_corr = system_level_correlation(p, h, s)

        def _spearman_fn(pred, human, doc_ids):
            return summary_level_correlation(pred, human, doc_ids)["spearman"]

        pt, lo, hi = bootstrap_ci(p, h, d, _spearman_fn, n_resamples=n_bootstrap, seed=seed)

        rows.append({
            "metric": col,
            "sample_spearman": sample_corr["spearman"],
            "sample_kendall": sample_corr["kendall"],
            "summary_spearman": summary_corr["spearman"],
            "summary_spearman_ci_lo": lo,
            "summary_spearman_ci_hi": hi,
            "summary_kendall": summary_corr["kendall"],
            "system_spearman": system_corr["spearman"],
        })

    result = pd.DataFrame(rows).sort_values("summary_spearman", ascending=False)
    return result
