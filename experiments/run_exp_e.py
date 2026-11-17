"""
experiments/run_exp_e.py
Experiment E: Correlation matrix + heatmaps + bootstrap CIs.
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import (
    load_summeval, records_to_dataframe, load_config, set_random_seed,
    ensure_results_dir, save_results, DIMENSIONS, HUMAN_DIMENSIONS,
    summary_level_correlation, pooled_correlation, system_level_correlation,
    top_k_overlap, bootstrap_ci,
)
from src.traditional_metrics import compute_all_traditional_metrics, TRADITIONAL_METRIC_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path, is_mock: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(matrix.columns) * 1.5), max(6, len(matrix) * 0.5)))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.3)
    if is_mock:
        title += " [MOCK DATA]"
    ax.set_title(title, fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved heatmap: {output_path}")


def run_experiment_e(
    limit: int | None = None,
    mock: bool = False,
    results_dir: Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    cfg = load_config()
    set_random_seed(seed)
    res_dir = results_dir or (
        Path(cfg["mock_results_dir"]) if mock else Path(cfg["results_dir"])
    )
    ensure_results_dir(res_dir)
    ensure_results_dir(res_dir / "figures")

    records = load_summeval()
    human_df = records_to_dataframe(records)

    trad_df = compute_all_traditional_metrics(
        records, limit=limit, results_dir=res_dir,
        output_path=res_dir / "traditional_metrics.csv",
    )

    # Try to load G-Eval scores
    suffix = "_mock" if mock else ""
    geval_path = res_dir / f"geval_scores_full{suffix}.csv"
    geval_df = None
    if geval_path.exists():
        geval_df = pd.read_csv(geval_path)
        logger.info(f"Loaded G-Eval scores from {geval_path}")
    else:
        logger.warning(f"G-Eval scores not found at {geval_path}. Run run_exp_a.py first.")

    combined = human_df[["doc_id", "system_id"] + [f"human_{d}" for d in HUMAN_DIMENSIONS]].copy()
    trad_no_human = trad_df.drop(
        columns=[c for c in trad_df.columns if c.startswith("human_")], errors="ignore"
    )
    combined = combined.merge(trad_no_human, on=["doc_id", "system_id"], how="left")
    if limit:
        combined = combined.iloc[:limit]

    metric_cols = list(TRADITIONAL_METRIC_COLS)
    if geval_df is not None:
        for dim in list(DIMENSIONS) + ["overall"]:
            dim_scores = geval_df[geval_df["dimension"] == dim][["doc_id", "system_id", "weighted_score"]]
            dim_scores = dim_scores.rename(columns={"weighted_score": f"geval_{dim}"})
            combined = combined.merge(dim_scores, on=["doc_id", "system_id"], how="left")
            metric_cols.append(f"geval_{dim}")

    corr_rows = []
    doc_ids = combined["doc_id"].values
    system_ids = combined["system_id"].values

    for metric_col in metric_cols:
        if metric_col not in combined.columns:
            continue
        pred = combined[metric_col].values.astype(float)

        for human_dim in HUMAN_DIMENSIONS:
            human_col = f"human_{human_dim}"
            human = combined[human_col].values.astype(float)
            valid = ~(np.isnan(pred) | np.isnan(human))
            if valid.sum() < 10:
                continue
            p, h, d, s = pred[valid], human[valid], doc_ids[valid], system_ids[valid]

            sum_corr = summary_level_correlation(p, h, d)
            samp_corr = pooled_correlation(p, h)
            sys_corr = system_level_correlation(p, h, s)

            def _spearman_fn(pred, human, doc_ids):
                return summary_level_correlation(pred, human, doc_ids)["spearman"]
            _, lo, hi = bootstrap_ci(p, h, d, _spearman_fn, n_resamples=1000, seed=seed)
            topk = top_k_overlap(p, h, s, k=5)

            corr_rows.append({
                "metric": metric_col, "human_dim": human_dim,
                "summary_spearman": sum_corr["spearman"],
                "summary_spearman_ci_lo": lo, "summary_spearman_ci_hi": hi,
                "summary_kendall": sum_corr["kendall"],
                "summary_pearson": sum_corr["pearson"],
                "sample_spearman": samp_corr["spearman"],
                "sample_kendall": samp_corr["kendall"],
                "system_spearman": sys_corr["spearman"],
                "system_kendall": sys_corr["kendall"],
                "top5_overlap": topk,
            })

    corr_df = pd.DataFrame(corr_rows)
    save_results(corr_df, res_dir / f"correlations{suffix}.csv")

    if not corr_df.empty:
        for corr_type, col in [("Summary-Level Spearman", "summary_spearman"),
                                ("System-Level Spearman", "system_spearman")]:
            try:
                pivot = corr_df.pivot_table(index="metric", columns="human_dim", values=col)
                plot_heatmap(pivot, f"{corr_type} Correlation",
                             res_dir / "figures" / f"heatmap_{col}{suffix}.png", is_mock=mock)
            except Exception as e:
                logger.warning(f"Heatmap failed: {e}")

    print("\n=== Experiment E: Correlations saved ===")
    return corr_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment E: Correlation Matrix")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment_e(limit=args.limit, mock=args.mock, results_dir=args.results_dir, seed=args.seed)
