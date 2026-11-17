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

from src.utils import (
    PROJECT_ROOT,
    DIMENSIONS,
    ensure_results_dir,
    records_to_dataframe,
    save_results,
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

    Stratifies by system_id and human overall score tertile to ensure
    balanced coverage.

    Args:
        df:        Full SummEval DataFrame (from records_to_dataframe).
        n_samples: Target number of records.
        seed:      Random seed.

    Returns:
        Stratified subset DataFrame.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Create tertile bins of human overall score
    df["overall_tertile"] = pd.qcut(
        df["human_overall"], q=3, labels=["low", "mid", "high"]
    )
    df["stratum"] = df["system_id"].astype(str) + "_" + df["overall_tertile"].astype(str)

    # Sample proportionally from each stratum
    strata = df["stratum"].unique()
    n_per_stratum = max(1, n_samples // len(strata))
    sampled = []
    for stratum in strata:
        stratum_df = df[df["stratum"] == stratum]
        k = min(n_per_stratum, len(stratum_df))
        idx = rng.choice(len(stratum_df), size=k, replace=False)
        sampled.append(stratum_df.iloc[idx])

    subset = pd.concat(sampled, ignore_index=True)
    # Top up if short
    if len(subset) < n_samples:
        remaining = df[~df.index.isin(subset.index)]
        extra_idx = rng.choice(len(remaining), size=n_samples - len(subset), replace=False)
        subset = pd.concat([subset, remaining.iloc[extra_idx]], ignore_index=True)

    subset = subset.sample(frac=1, random_state=seed).reset_index(drop=True)
    return subset.drop(columns=["stratum", "overall_tertile"], errors="ignore")


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
        logger.warning(f"Human ratings file missing columns: {missing}")
        return None

    # Drop unrated rows
    for d in DIMENSIONS:
        my_ratings = my_ratings[pd.notna(my_ratings[f"rated_{d}"])]
        my_ratings = my_ratings[my_ratings[f"rated_{d}"] != ""]

    if len(my_ratings) == 0:
        logger.warning("No completed ratings found in human_ratings.csv")
        return None

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
        "n_records": len(merged),
        "dimensions": {},
    }

    for dim in DIMENSIONS:
        my_col = f"rated_{dim}"
        expert_col = f"human_{dim}"
        my_scores = pd.to_numeric(merged[my_col], errors="coerce").dropna()
        expert_scores = pd.to_numeric(merged[expert_col], errors="coerce").dropna()

        # Align
        common_idx = my_scores.index.intersection(expert_scores.index)
        my_scores = my_scores[common_idx].values
        expert_scores = expert_scores[common_idx].values

        if len(my_scores) < 3:
            results["dimensions"][dim] = {"spearman": float("nan"), "alpha": float("nan"), "n": len(my_scores)}
            continue

        spearman_r, spearman_p = stats.spearmanr(my_scores, expert_scores)

        # Krippendorff alpha
        try:
            import krippendorff
            # Format: (n_raters, n_items)
            data = np.array([my_scores, expert_scores])
            alpha = krippendorff.alpha(data, level_of_measurement="ordinal")
        except ImportError:
            logger.warning("krippendorff package not installed; skipping alpha computation.")
            alpha = float("nan")
        except Exception as e:
            logger.warning(f"Krippendorff alpha failed for {dim}: {e}")
            alpha = float("nan")

        results["dimensions"][dim] = {
            "spearman": float(spearman_r),
            "spearman_p": float(spearman_p),
            "krippendorff_alpha": float(alpha),
            "n": int(len(my_scores)),
        }
        logger.info(
            f"  {dim}: Spearman={spearman_r:.3f} (p={spearman_p:.4f}), "
            f"Krippendorff alpha={alpha:.3f}, n={len(my_scores)}"
        )

    return results


def run_experiment_d(
    records: List[Dict[str, Any]],
    n_samples: int = 160,
    seed: int = 42,
    results_dir: Optional[Path] = None,
    human_ratings_path: Optional[Path] = None,
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
    template_path = export_rating_template(subset)

    agreement = compute_inter_rater_agreement(df, human_ratings_path=human_ratings_path)

    output = {
        "subset_size": len(subset),
        "template_path": str(template_path),
        "inter_rater_agreement": agreement,
    }

    # Save agreement report
    if results_dir and agreement:
        ensure_results_dir(results_dir)
        agreement_path = results_dir / "inter_rater_agreement.json"
        import json
        with open(agreement_path, "w") as f:
            json.dump(agreement, f, indent=2)
        logger.info(f"Inter-rater agreement saved to {agreement_path}")

    return output
