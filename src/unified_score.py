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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from src.utils import (
    minmax_normalize,
    summary_level_correlation,
    pooled_correlation,
    system_level_correlation,
    bootstrap_summary_level_ci,
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
    raw_components = _raw_components(
        df,
        geval_col=geval_col,
        bertscore_col=bertscore_col,
        rouge_col=rouge_col,
        ensemble_std_col=ensemble_std_col,
    )
    return tuple(minmax_normalize(values) for values in raw_components)


def _raw_components(
    df: pd.DataFrame,
    geval_col: str,
    bertscore_col: str,
    rouge_col: str,
    ensemble_std_col: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return required, unnormalized score components without silent defaults."""
    columns = [geval_col, bertscore_col, rouge_col]
    if ensemble_std_col is None:
        raise ValueError("An ensemble standard-deviation column is required")
    columns.append(ensemble_std_col)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Unified score is missing required metric columns: {missing}")

    geval, bertscore, rouge, judge_std = (
        df[column].to_numpy(dtype=float) for column in columns
    )
    components = (geval, bertscore, rouge, 1.0 - judge_std)
    if any(not np.isfinite(values).all() for values in components):
        raise ValueError("Unified score components must contain only finite values")
    return components


def _scale_fold(
    train: np.ndarray,
    validation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale features using training-fold bounds only."""
    minimum = train.min(axis=0)
    span = train.max(axis=0) - minimum
    safe_span = np.where(span == 0, 1.0, span)
    train_scaled = (train - minimum) / safe_span
    validation_scaled = (validation - minimum) / safe_span
    train_scaled[:, span == 0] = 0.0
    validation_scaled[:, span == 0] = 0.0
    return train_scaled, validation_scaled


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
    weight_values = np.asarray(list(weights.values()), dtype=float)
    if (
        not np.isfinite(weight_values).all()
        or np.any(weight_values < 0)
        or not np.isclose(weight_values.sum(), 1.0)
    ):
        raise ValueError("Unified-score weights must be finite, non-negative, and sum to 1")
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
    n_folds: int = 5,
    **extract_kwargs: Any,
) -> pd.DataFrame:
    """Grid search over the weight simplex using document-grouped CV.

    Args:
        df:         DataFrame with all components and human_col.
        human_col:  Human ground truth column.
        step:       Grid step size (0.05 → 1771 valid combinations).

    Returns:
        DataFrame of all weight combinations sorted by Spearman descending.
    """
    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1]")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    raw_components = _raw_components(
        df,
        geval_col=extract_kwargs.get("geval_col", "geval_overall"),
        bertscore_col=extract_kwargs.get("bertscore_col", "bertscore_F1_ref"),
        rouge_col=extract_kwargs.get("rouge_col", "rouge1_ref"),
        ensemble_std_col=extract_kwargs.get("ensemble_std_col", "geval_judge_std"),
    )
    X_all = np.column_stack(raw_components)
    human = df[human_col].values.astype(float)
    doc_ids = df[doc_id_col].values
    valid = np.isfinite(human)
    X = X_all[valid]
    y = human[valid]
    groups = doc_ids[valid]
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("Grid search requires at least two unique documents")
    splitter = GroupKFold(n_splits=min(n_folds, n_groups))
    folds = list(splitter.split(X, y, groups=groups))

    vals = np.unique(np.append(np.arange(0, 1 + step / 2, step), 1.0))
    rows = []
    for w_g, w_b, w_r in product(vals, vals, vals):
        w_a = 1.0 - w_g - w_b - w_r
        if w_a < -1e-9 or w_a > 1 + 1e-9:
            continue
        w_a = max(0.0, min(1.0, w_a))

        weights = np.array([w_g, w_b, w_r, w_a])
        fold_correlations = []
        for train_idx, validation_idx in folds:
            _, X_val = _scale_fold(X[train_idx], X[validation_idx])
            predictions = X_val @ weights
            fold_correlations.append(
                summary_level_correlation(
                    predictions,
                    y[validation_idx],
                    groups[validation_idx],
                )
            )
        rows.append({
            "w_geval": round(w_g, 4),
            "w_bertscore": round(w_b, 4),
            "w_rouge": round(w_r, 4),
            "w_ensemble": round(w_a, 4),
            "spearman": float(np.nanmean([c["spearman"] for c in fold_correlations])),
            "kendall": float(np.nanmean([c["kendall"] for c in fold_correlations])),
            "pearson": float(np.nanmean([c["pearson"] for c in fold_correlations])),
            "cv_folds": len(folds),
        })

    result = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    return result


def grid_weight_oof_predictions(
    df: pd.DataFrame,
    weights: Dict[str, float],
    human_col: str,
    doc_id_col: str = "doc_id",
    n_folds: int = 5,
    **extract_kwargs: Any,
) -> np.ndarray:
    """Apply selected grid weights out of fold with train-only feature scaling."""
    weight_vector = np.asarray(
        [
            weights["geval"],
            weights["bertscore"],
            weights["rouge"],
            weights["ensemble_agreement"],
        ],
        dtype=float,
    )
    if (
        not np.isfinite(weight_vector).all()
        or np.any(weight_vector < 0)
        or not np.isclose(weight_vector.sum(), 1.0)
    ):
        raise ValueError("Grid weights must be finite, non-negative, and sum to 1")
    raw_components = _raw_components(
        df,
        geval_col=extract_kwargs.get("geval_col", "geval_overall"),
        bertscore_col=extract_kwargs.get("bertscore_col", "bertscore_F1_ref"),
        rouge_col=extract_kwargs.get("rouge_col", "rouge1_ref"),
        ensemble_std_col=extract_kwargs.get("ensemble_std_col", "geval_judge_std"),
    )
    X_all = np.column_stack(raw_components)
    human = df[human_col].to_numpy(dtype=float)
    doc_ids = df[doc_id_col].to_numpy()
    valid = np.isfinite(human)
    X = X_all[valid]
    y = human[valid]
    groups = doc_ids[valid]
    n_groups = len(np.unique(groups))
    if n_folds < 2 or n_groups < 2:
        raise ValueError("Grid-weight predictions require at least two folds and documents")

    predictions = np.full(len(df), np.nan)
    splitter = GroupKFold(n_splits=min(n_folds, n_groups))
    valid_indices = np.flatnonzero(valid)
    for train_idx, validation_idx in splitter.split(X, y, groups=groups):
        _, X_validation = _scale_fold(X[train_idx], X[validation_idx])
        predictions[valid_indices[validation_idx]] = X_validation @ weight_vector
    return predictions


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
    raw_components = _raw_components(
        df,
        geval_col=extract_kwargs.get("geval_col", "geval_overall"),
        bertscore_col=extract_kwargs.get("bertscore_col", "bertscore_F1_ref"),
        rouge_col=extract_kwargs.get("rouge_col", "rouge1_ref"),
        ensemble_std_col=extract_kwargs.get("ensemble_std_col", "geval_judge_std"),
    )
    X_all = np.column_stack(raw_components)
    human = df[human_col].values.astype(float)
    doc_ids = df[doc_id_col].values

    # Remove NaN targets
    valid = ~np.isnan(human)
    X = X_all[valid]
    y = human[valid]
    groups = doc_ids[valid]

    n_groups = len(np.unique(groups))
    if n_folds < 2 or n_groups < 2:
        raise ValueError("Learned combination requires at least two folds and documents")
    actual_folds = min(n_folds, n_groups)
    gkf = GroupKFold(n_splits=actual_folds)
    oof_preds = np.full(len(y), float("nan"))
    fold_metrics = []
    all_coefs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_val = _scale_fold(X[train_idx], X[val_idx])
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

        rho = summary_level_correlation(
            preds,
            y_val,
            groups[val_idx],
        )["spearman"]
        fold_metrics.append({"fold": fold, "spearman": float(rho)})

    mean_spearman = float(np.nanmean([m["spearman"] for m in fold_metrics]))
    mean_coefs = np.mean(all_coefs, axis=0)

    oof_full = np.full(len(df), float("nan"))
    oof_full[valid] = oof_preds
    return {
        "model_type": "gradient_boosting" if use_gradient_boosting else "ridge",
        "cv_folds": actual_folds,
        "fold_metrics": fold_metrics,
        "mean_cv_spearman": mean_spearman,
        "feature_names": ["geval", "bertscore", "rouge", "ensemble_agreement"],
        "mean_feature_importances": mean_coefs.tolist(),
        "oof_predictions": oof_full.tolist(),
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

    all_metrics = list(dict.fromkeys([*metric_cols, unified_col]))
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

        _, lo, hi = bootstrap_summary_level_ci(
            p,
            h,
            d,
            statistic="spearman",
            n_resamples=n_bootstrap,
            seed=seed,
        )

        rows.append({
            "metric": col,
            "sample_spearman": sample_corr["spearman"],
            "sample_kendall": sample_corr["kendall"],
            "sample_pearson": sample_corr["pearson"],
            "summary_spearman": summary_corr["spearman"],
            "summary_spearman_ci_lo": lo,
            "summary_spearman_ci_hi": hi,
            "summary_kendall": summary_corr["kendall"],
            "summary_pearson": summary_corr["pearson"],
            "system_spearman": system_corr["spearman"],
            "system_kendall": system_corr["kendall"],
            "system_pearson": system_corr["pearson"],
        })

    result = pd.DataFrame(rows).sort_values("summary_spearman", ascending=False)
    return result
