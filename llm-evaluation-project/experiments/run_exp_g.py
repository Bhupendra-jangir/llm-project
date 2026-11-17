"""
experiments/run_exp_g.py
Experiment G: Unified score.
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
    ensure_results_dir, save_results, DIMENSIONS,
)
from src.traditional_metrics import compute_all_traditional_metrics
from src.unified_score import (
    compute_unified_score, grid_search_weights, learned_combiner,
    compare_all_metrics, DEFAULT_WEIGHTS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_experiment_g(
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
    combined = trad_df.copy()
    if limit:
        combined = combined.iloc[:limit]

    suffix = "_mock" if mock else ""
    geval_path = res_dir / f"geval_scores_full{suffix}.csv"
    if geval_path.exists():
        geval_df = pd.read_csv(geval_path)
        overall = geval_df[geval_df["dimension"] == "overall"][["doc_id", "system_id", "weighted_score"]]
        overall = overall.rename(columns={"weighted_score": "geval_overall"})
        combined = combined.merge(overall, on=["doc_id", "system_id"], how="left")

        judge_path = res_dir / f"exp_c_judge_scores{suffix}.csv"
        if judge_path.exists():
            judge_df = pd.read_csv(judge_path)
            overall_j = judge_df[judge_df["dimension"] == "overall"].copy()
            judge_std = overall_j.groupby(["doc_id", "system_id"])["weighted_score"].std().reset_index()
            judge_std.columns = ["doc_id", "system_id", "geval_judge_std"]
            combined = combined.merge(judge_std, on=["doc_id", "system_id"], how="left")

    target_human = "human_overall"

    # (a) Fixed weights
    combined["unified_fixed"] = compute_unified_score(combined, weights=DEFAULT_WEIGHTS)

    # (b) Grid search
    step = cfg.get("unified_score", {}).get("grid_search_step", 0.05)
    logger.info(f"Running weight grid search (step={step})...")
    grid_results = grid_search_weights(combined, human_col=target_human, step=step)
    best = grid_results.iloc[0]
    logger.info(f"Best grid weights: {best.to_dict()}")
    save_results(grid_results.head(100), res_dir / f"exp_g_grid_search{suffix}.csv")

    best_w = {"geval": best.w_geval, "bertscore": best.w_bertscore,
               "rouge": best.w_rouge, "ensemble_agreement": best.w_ensemble}
    combined["unified_grid"] = compute_unified_score(combined, weights=best_w)

    # (c) Ridge regression
    ridge_result = learned_combiner(combined, human_col=target_human, seed=seed)
    oof = ridge_result["oof_predictions"]
    combined["unified_ridge"] = np.array(oof[:len(combined)], dtype=float)

    metric_cols = ["rouge1_ref", "rouge2_ref", "bertscore_F1_ref", "bleu_ref",
                   "unified_fixed", "unified_grid", "unified_ridge"]
    if "geval_overall" in combined.columns:
        metric_cols.insert(0, "geval_overall")

    comparison = compare_all_metrics(
        combined, human_col=target_human, metric_cols=metric_cols,
        unified_col="unified_fixed", seed=seed,
    )
    save_results(comparison, res_dir / f"exp_g_comparison{suffix}.csv")

    print("\n=== Experiment G: Unified Score Comparison ===")
    display_cols = ["metric", "summary_spearman", "summary_spearman_ci_lo",
                    "summary_spearman_ci_hi", "system_spearman"]
    display_cols = [c for c in display_cols if c in comparison.columns]
    print(comparison[display_cols].to_string(index=False))
    print(f"\nRidge CV Spearman: {ridge_result['mean_cv_spearman']:.4f}")
    print("Feature importances:",
          dict(zip(ridge_result["feature_names"],
                   [f"{x:.3f}" for x in ridge_result["mean_feature_importances"]])))

    print("\n=== RESEARCH QUESTION ANSWER ===")
    best_unified = comparison[comparison["metric"].str.startswith("unified")]["summary_spearman"].max()
    best_individual = comparison[~comparison["metric"].str.startswith("unified")]["summary_spearman"].max()
    is_better = best_unified > best_individual
    print(f"Best unified score Spearman:    {best_unified:.4f}")
    print(f"Best individual metric Spearman: {best_individual:.4f}")
    print(f"Unified score is {'BETTER' if is_better else 'NOT BETTER'} than the best individual metric.")
    if mock:
        print("[NOTE: Results based on MOCK DATA — not meaningful for research conclusions]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment G: Unified Score")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment_g(limit=args.limit, mock=args.mock, results_dir=args.results_dir, seed=args.seed)
