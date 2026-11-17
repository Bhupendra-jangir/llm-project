"""
experiments/run_exp_f.py
Experiment F: Evaluator bias analysis.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.utils import (
    load_summeval, load_config, set_random_seed,
    ensure_results_dir, DIMENSIONS,
)
from src.traditional_metrics import compute_all_traditional_metrics
from src.bias_analysis import run_bias_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_experiment_f(
    limit: int | None = None,
    mock: bool = False,
    results_dir: Path | None = None,
    seed: int = 42,
) -> None:
    cfg = load_config()
    set_random_seed(seed)
    res_dir = results_dir or (
        Path(cfg["mock_results_dir"]) if mock else Path(cfg["results_dir"])
    )
    ensure_results_dir(res_dir)

    records = load_summeval()
    trad_df = compute_all_traditional_metrics(
        records, limit=limit, results_dir=res_dir,
        output_path=res_dir / "traditional_metrics.csv",
    )

    # Merge text columns
    text_df = pd.DataFrame([{
        "doc_id": r["doc_id"], "system_id": r["system_id"],
        "system_output": r["system_output"], "source": r["source"]
    } for r in records])
    combined = trad_df.merge(text_df, on=["doc_id", "system_id"], how="left")
    if limit:
        combined = combined.iloc[:limit]

    # Add G-Eval if available
    suffix = "_mock" if mock else ""
    geval_path = res_dir / f"geval_scores_full{suffix}.csv"
    if geval_path.exists():
        geval_df = pd.read_csv(geval_path)
        for dim in DIMENSIONS:
            dim_scores = geval_df[geval_df["dimension"] == dim][["doc_id", "system_id", "weighted_score"]]
            dim_scores = dim_scores.rename(columns={"weighted_score": f"geval_{dim}"})
            combined = combined.merge(dim_scores, on=["doc_id", "system_id"], how="left")

    metric_human_pairs = [
        ("rouge1_ref", "relevance"),
        ("bertscore_F1_ref", "consistency"),
    ]
    for dim in DIMENSIONS:
        col = f"geval_{dim}"
        if col in combined.columns:
            metric_human_pairs.append((col, dim))

    all_results = []
    for metric_col, human_dim in metric_human_pairs:
        if metric_col not in combined.columns:
            continue
        logger.info(f"Bias analysis: {metric_col} vs human_{human_dim}")
        result = run_bias_analysis(
            df=combined, metric_col=metric_col,
            human_col=f"human_{human_dim}", metric_name=metric_col,
            results_dir=res_dir, is_mock=mock,
        )
        all_results.append(result)

    with open(res_dir / f"exp_f_bias_summary{suffix}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Bias analysis complete. Results in {res_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment F: Bias Analysis")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    run_experiment_f(limit=args.limit, mock=args.mock, results_dir=args.results_dir)
