"""
src/human_alignment.py
=======================
Human validation subset creation, rating UI support, and inter-rater agreement.

Implements Experiment D:
  - Stratified sampling (by system_id and human overall score tertile)
  - Export dashboard/human_rating_template.csv
  - Compute inter-rater agreement (Spearman + Krippendorff alpha)
    when data/human_ratings.csv exists
  - Code runs correctly whether or not human_ratings.csv exists
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from src.utils import (
    PROJECT_ROOT,
    DIMENSIONS,
    HUMAN_DIMENSIONS,
    ensure_results_dir,
    records_to_dataframe,
    save_results,
    summary_level_correlation,
)

logger = logging.getLogger(__name__)

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Stratified subset creation
# ---------------------------------------------------------------------------

def create_stratified_subset(
    df: pd.DataFrame,
    n_samples: int = 160,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a stratified subset for human validation.

    Stratifies by system_id and human overall score quartile to ensure
    balanced coverage.

    Args:
        df:        Full SummEval DataFrame (from records_to_dataframe).
        n_samples: Target number of records.
        seed:      Random seed.

    Returns:
        Stratified subset DataFrame.
    """
    if not 0 < n_samples <= len(df):
        raise ValueError(f"n_samples must be between 1 and {len(df)}")
    rng = np.random.default_rng(seed)
    sampled_df = df.reset_index(drop=True).copy()
    sampled_df["score_quartile"] = (
        pd.qcut(
            sampled_df["human_overall"].rank(method="first"),
            q=4,
            labels=False,
        )
        + 1
    )
    grouped = {
        key: group.index.to_numpy()
        for key, group in sampled_df.groupby(["system_id", "score_quartile"], sort=True)
    }
    sizes = {key: len(indices) for key, indices in grouped.items()}
    base_count, _ = divmod(n_samples, len(grouped))
    allocations = {
        key: min(base_count, size) for key, size in sizes.items()
    }
    remaining = n_samples - sum(allocations.values())
    priority = list(grouped)
    rng.shuffle(priority)
    while remaining:
        progressed = False
        for key in priority:
            if allocations[key] < sizes[key]:
                allocations[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Unable to allocate the requested stratified sample")

    selected_indices = []
    for key, indices in grouped.items():
        count = allocations[key]
        if count:
            selected_indices.extend(rng.choice(indices, size=count, replace=False))
    return (
        sampled_df.loc[selected_indices]
        .sample(frac=1, random_state=seed)
        .drop(columns="score_quartile")
        .reset_index(drop=True)
    )


def export_rating_template(
    subset: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Export a CSV template for human ratings.

    Columns: id, doc_id, system_id, source, reference, system_output,
    rated_coherence, rated_consistency, rated_fluency, rated_relevance
    (the 'rated_*' columns are blank for the human to fill in).

    Args:
        subset:      Stratified subset DataFrame.
        output_path: Where to save the template.

    Returns:
        Path where the template was saved.
    """
    path = output_path or DASHBOARD_DIR / "human_rating_template.csv"
    ensure_results_dir(path.parent)
    template = subset[["doc_id", "system_id", "source", "reference", "system_output"]].copy()
    template.insert(0, "id", range(1, len(template) + 1))
    for dim in DIMENSIONS:
        template[f"rated_{dim}"] = ""  # Blank for human to fill

    save_results(template, path)
    logger.info(f"Human rating template exported to {path} ({len(template)} records)")
    return path


# ---------------------------------------------------------------------------
# Inter-rater agreement
# ---------------------------------------------------------------------------

def compute_inter_rater_agreement(
    summeval_df: pd.DataFrame,
    human_ratings_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Compute inter-rater agreement between manual ratings and SummEval experts.

    Returns None if human_ratings.csv doesn't exist (graceful degradation).

    Args:
        summeval_df:        Full SummEval DataFrame.
        human_ratings_path: Path to data/human_ratings.csv.

    Returns:
        Dict with Spearman rho and Krippendorff alpha per dimension, or None.
    """
    path = human_ratings_path or DATA_DIR / "human_ratings.csv"
    if not path.exists():
        logger.info(
            f"Human ratings file not found at {path}. "
            f"Skipping inter-rater agreement computation. "
            f"To enable: fill out dashboard/human_rating_template.csv and "
            f"save as data/human_ratings.csv."
        )
        return None

    logger.info(f"Loading human ratings from {path}")
    my_ratings = pd.read_csv(path)

    # Validate columns
    required = ["doc_id", "system_id"] + [f"rated_{d}" for d in DIMENSIONS]
    missing = [c for c in required if c not in my_ratings.columns]
    if missing:
        raise ValueError(f"Human ratings file missing columns: {missing}")

    # Merge with SummEval expert scores
    merged = my_ratings.merge(
        summeval_df[["doc_id", "system_id"] + [f"human_{d}" for d in DIMENSIONS]],
        on=["doc_id", "system_id"],
        how="inner",
    )

    if len(merged) == 0:
        logger.warning("No matching records between human ratings and SummEval.")
        return None

    results: Dict[str, Any] = {
        "n_records": 0,
        "dimensions": {},
    }

    for dim in DIMENSIONS:
        my_col = f"rated_{dim}"
        expert_col = f"human_{dim}"
        my_scores = pd.to_numeric(merged[my_col], errors="coerce")
        expert_scores = pd.to_numeric(merged[expert_col], errors="coerce")
        valid = my_scores.between(1, 5) & expert_scores.notna()
        my_scores = my_scores[valid].to_numpy(dtype=float)
        expert_scores = expert_scores[valid].to_numpy(dtype=float)

        if len(my_scores) < 3:
            results["dimensions"][dim] = {
                "spearman": float("nan"),
                "cohen_kappa": float("nan"),
                "weighted_cohen_kappa": float("nan"),
                "krippendorff_alpha": float("nan"),
                "n": len(my_scores),
            }
            continue

        spearman_r, spearman_p = stats.spearmanr(my_scores, expert_scores)
        rated_categories = np.rint(my_scores).astype(int)
        expert_categories = np.rint(expert_scores).astype(int)
        cohen_kappa = cohen_kappa_score(rated_categories, expert_categories)
        weighted_kappa = cohen_kappa_score(
            rated_categories,
            expert_categories,
            weights="linear",
        )
        import krippendorff

        alpha = krippendorff.alpha(
            np.array([rated_categories, expert_categories]),
            level_of_measurement="ordinal",
        )
        results["n_records"] = max(results["n_records"], len(my_scores))

        results["dimensions"][dim] = {
            "spearman": float(spearman_r),
            "spearman_p": float(spearman_p),
            "cohen_kappa": float(cohen_kappa),
            "weighted_cohen_kappa": float(weighted_kappa),
            "krippendorff_alpha": float(alpha),
            "n": int(len(my_scores)),
        }
        logger.info(
            f"  {dim}: Spearman={spearman_r:.3f} (p={spearman_p:.4f}), "
            f"Cohen kappa={cohen_kappa:.3f}, weighted kappa={weighted_kappa:.3f}, "
            f"Krippendorff alpha={alpha:.3f}, n={len(my_scores)}"
        )

    return results


def run_experiment_d(
    records: List[Dict[str, Any]],
    n_samples: int = 160,
    seed: int = 42,
    results_dir: Optional[Path] = None,
    human_ratings_path: Optional[Path] = None,
    automatic_metrics: Optional[pd.DataFrame] = None,
    rating_sheet_path: Optional[Path] = None,
    data_label: str = "REAL DATA",
) -> Dict[str, Any]:
    """Run Experiment D: human validation subset and inter-rater agreement.

    Args:
        records:            SummEval records.
        n_samples:          Subset size.
        seed:               Random seed.
        results_dir:        Output directory.
        human_ratings_path: Path to data/human_ratings.csv (optional).

    Returns:
        Dict with subset size, template path, and agreement results.
    """
    df = records_to_dataframe(records)
    subset = create_stratified_subset(df, n_samples=n_samples, seed=seed)
    template_path = export_rating_template(subset, output_path=rating_sheet_path)

    agreement = compute_inter_rater_agreement(df, human_ratings_path=human_ratings_path)

    alignment = None
    if automatic_metrics is not None:
        alignment = compute_metric_alignment(automatic_metrics, df)
        alignment["data_label"] = data_label

    output = {
        "subset_size": len(subset),
        "template_path": str(template_path),
        "inter_rater_agreement": agreement,
        "automatic_metric_alignment": (
            alignment.to_dict(orient="records") if alignment is not None else None
        ),
        "human_reference": (
            "new human ratings compared with SummEval expert annotations"
            if agreement is not None
            else "SummEval expert annotations only; no new human ratings were present"
        ),
        "data_label": data_label,
    }

    if results_dir:
        ensure_results_dir(results_dir)
        agreement_path = results_dir / "exp_d_human_alignment.json"
        import json
        with agreement_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Human alignment report saved to {agreement_path}")
        if alignment is not None:
            save_results(alignment, results_dir / "exp_d_metric_alignment.csv")

    return output


def compute_metric_alignment(
    automatic_metrics: pd.DataFrame,
    expert_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Correlate available automatic metric columns with SummEval expert scores."""
    nonhuman_metrics = automatic_metrics.drop(
        columns=[column for column in automatic_metrics if column.startswith("human_")],
        errors="ignore",
    )
    merged = nonhuman_metrics.merge(
        expert_scores[
            ["doc_id", "system_id", *[f"human_{dim}" for dim in HUMAN_DIMENSIONS]]
        ],
        on=["doc_id", "system_id"],
        how="inner",
        suffixes=("", "_expert"),
    )
    metric_columns = [
        column
        for column in automatic_metrics.columns
        if column not in {"doc_id", "system_id", "bertscore_model"}
        and not column.startswith("human_")
        and pd.api.types.is_numeric_dtype(automatic_metrics[column])
    ]
    rows = []
    for metric_col in metric_columns:
        for dimension in HUMAN_DIMENSIONS:
            human_col = f"human_{dimension}"
            valid = merged[[metric_col, human_col]].notna().all(axis=1)
            if valid.sum() < 3:
                continue
            pred = merged.loc[valid, metric_col].to_numpy(dtype=float)
            human = merged.loc[valid, human_col].to_numpy(dtype=float)
            docs = merged.loc[valid, "doc_id"].to_numpy()
            summary_corr = summary_level_correlation(pred, human, docs)
            system_pred = merged.loc[valid].groupby("system_id")[metric_col].mean()
            system_human = merged.loc[valid].groupby("system_id")[human_col].mean()
            rows.append(
                {
                    "metric": metric_col,
                    "human_dim": dimension,
                    "summary_spearman": summary_corr["spearman"],
                    "summary_pearson": summary_corr["pearson"],
                    "summary_kendall": summary_corr["kendall"],
                    "pooled_spearman": float(stats.spearmanr(pred, human).statistic),
                    "pooled_pearson": float(stats.pearsonr(pred, human).statistic),
                    "pooled_kendall": float(stats.kendalltau(pred, human).statistic),
                    "system_spearman": float(
                        stats.spearmanr(system_pred, system_human).statistic
                    ),
                    "system_pearson": float(
                        stats.pearsonr(system_pred, system_human).statistic
                    ),
                    "system_kendall": float(
                        stats.kendalltau(system_pred, system_human).statistic
                    ),
                }
            )
    return pd.DataFrame(rows)
