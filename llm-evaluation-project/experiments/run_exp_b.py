"""
experiments/run_exp_b.py
========================
Experiment B: ROUGE, BLEU, and BERTScore on SummEval.
Computes all traditional metrics and correlates with every human dimension.
Outputs: results/traditional_metrics.csv, results/exp_b_correlations.csv
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.utils import (
    load_summeval, records_to_dataframe, load_config, set_random_seed,
    ensure_results_dir, save_results, HUMAN_DIMENSIONS, DIMENSIONS,
    summary_level_correlation, pooled_correlation, system_level_correlation,
    bootstrap_ci,
)
from src.traditional_metrics import (
    compute_all_traditional_metrics, TRADITIONAL_METRIC_COLS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_experiment_b(
    limit: int | None = None,
    results_dir: Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run Experiment B: traditional metrics evaluation."""
    cfg = load_config()
    set_random_seed(seed)
    res_dir = results_dir or Path(cfg["results_dir"])
    ensure_results_dir(res_dir)

    logger.info("=== Experiment B: Traditional Metrics ===")
    records = load_summeval(limit=limit)
    logger.info(f"Loaded {len(records)} records.")

    # Compute or load metrics
    metrics_df = compute_all_traditional_metrics(
        records,
        limit=limit,
        results_dir=res_dir,
        output_path=res_dir / "traditional_metrics.csv",
    )
    logger.info(f"Metrics DataFrame shape: {metrics_df.shape}")

    # Correlate each metric vs each human dimension
    rows = []
    doc_ids = metrics_df["doc_id"].values
    system_ids = metrics_df["system_id"].values

    for metric_col in TRADITIONAL_METRIC_COLS:
        if metric_col not in metrics_df.columns:
            continue
        pred = metrics_df[metric_col].values.astype(float)

        for human_dim in HUMAN_DIMENSIONS:
            human_col = f"human_{human_dim}"
            if human_col not in metrics_df.columns:
                continue
            human = metrics_df[human_col].values.astype(float)

            valid = ~(np.isnan(pred) | np.isnan(human))
            if valid.sum() < 5:
                continue
            p, h, d, s = pred[valid], human[valid], doc_ids[valid], system_ids[valid]

            summary_corr = summary_level_correlation(p, h, d)
            sample_corr = pooled_correlation(p, h)
            system_corr = system_level_correlation(p, h, s)

            def _spearman(pred, human, doc_ids):
                return summary_level_correlation(pred, human, doc_ids)["spearman"]

            _, lo, hi = bootstrap_ci(p, h, d, _spearman, n_resamples=1000, seed=seed)

            rows.append({
                "metric": metric_col,
                "human_dim": human_dim,
                "summary_spearman": summary_corr["spearman"],
                "summary_spearman_ci_lo": lo,
                "summary_spearman_ci_hi": hi,
                "summary_kendall": summary_corr["kendall"],
                "summary_pearson": summary_corr["pearson"],
                "summary_n_docs": summary_corr["n_docs"],
                "sample_spearman": sample_corr["spearman"],
                "sample_kendall": sample_corr["kendall"],
                "system_spearman": system_corr["spearman"],
                "system_kendall": system_corr["kendall"],
            })

    corr_df = pd.DataFrame(rows)
    out_path = res_dir / "exp_b_correlations.csv"
    save_results(corr_df, out_path)
    logger.info(f"Saved correlations to {out_path}")

    # Print summary table
    print("\n=== Experiment B: Summary-Level Spearman Correlations (Traditional Metrics) ===")
    pivot = corr_df[corr_df["human_dim"].isin(["coherence", "consistency", "fluency", "relevance"])]
    if not pivot.empty:
        pivot_table = pivot.pivot_table(index="metric", columns="human_dim", values="summary_spearman")
        print(pivot_table.round(4).to_string())

    return corr_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment B: Traditional Metrics")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment_b(limit=args.limit, results_dir=args.results_dir, seed=args.seed)
