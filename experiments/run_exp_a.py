"""
experiments/run_exp_a.py
========================
Experiment A: G-Eval baseline on SummEval.
Compares with paper Table 1 (G-Eval-4 avg Spearman 0.514, Kendall 0.418).
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.utils import (
    load_summeval, records_to_dataframe, load_config, set_random_seed,
    ensure_results_dir, save_results, DIMENSIONS, HUMAN_DIMENSIONS,
    summary_level_correlation, pooled_correlation, system_level_correlation,
    bootstrap_summary_level_ci,
)
from src.llm_clients import create_client, check_api_key_available, DiskCache
from src.geval_evaluator import GEvalEvaluator, ABLATION_MODES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run_geval_ablation(
    ablation: str,
    client,
    records: list,
    limit: int | None,
    score_mode: str,
    n_samples: int,
    results_dir: Path,
) -> pd.DataFrame:
    """Run one G-Eval ablation variant."""
    evaluator = GEvalEvaluator(
        client=client,
        dimensions=list(DIMENSIONS),
        score_mode=score_mode,
        n_samples=n_samples,
        ablation=ablation,
        results_dir=results_dir,
    )
    df = await evaluator.evaluate_dataset(records, limit=limit, show_progress=True)
    df["ablation"] = ablation
    return df


def run_experiment_a(
    limit: int | None = None,
    mock: bool = False,
    results_dir: Path | None = None,
    ablations: list | None = None,
    seed: int = 42,
    records: list | None = None,
) -> None:
    """Run Experiment A: G-Eval baseline with ablation variants."""
    cfg = load_config()
    set_random_seed(seed)

    llm_cfg = cfg["llm"]
    provider = "mock" if mock else llm_cfg["provider"]
    if not mock and (provider == "mock" or not check_api_key_available(cfg)):
        print("\n" + "=" * 60)
        print("REAL EVALUATION NOT CONFIGURED")
        print("=" * 60)
        print("Continuing in mock mode. To run real G-Eval evaluations:")
        print("  1. Copy .env.example to .env")
        print("  2. Set your API key (e.g., OPENAI_API_KEY=sk-...)")
        print("  3. Update configs/config.yaml: set llm.provider and llm.model")
        print("  4. Quick test: make smoke-real   (runs --limit 20)")
        print("  5. Full run:   make all")
        print("=" * 60)
        provider = "mock"

    mock = provider == "mock"
    if results_dir is None:
        results_dir = (
            Path(cfg["mock_results_dir"]) if mock else Path(cfg["results_dir"])
        )
    ensure_results_dir(results_dir)

    model = "mock-model" if mock else llm_cfg["model"]
    cache = DiskCache(cfg["cache_db"]) if not mock else None

    client = create_client(
        provider=provider,
        model=model,
        cache=cache,
        concurrency_limit=cfg.get("concurrency_limit", 5),
        max_retries=cfg.get("max_retries", 5),
        retry_base_delay=cfg.get("retry_base_delay", 1.0),
        requests_per_minute=llm_cfg.get("requests_per_minute", 60),
    )

    score_mode = cfg["geval"]["score_mode"]
    n_samples = cfg["geval"]["n_samples"]
    ablation_list = ablations or cfg["geval"]["ablations"]

    logger.info("=== Experiment A: G-Eval Baseline ===")
    has_sampled_records = records is not None
    records = records if has_sampled_records else load_summeval()
    evaluation_limit = None if has_sampled_records and limit is None else limit

    # Print cost estimate
    dummy_ev = GEvalEvaluator(client=client, score_mode=score_mode, n_samples=n_samples)
    logger.info(dummy_ev.estimate_cost(min(limit or len(records), len(records))))

    async def evaluate_ablations() -> list[pd.DataFrame]:
        frames = []
        for ablation in ablation_list:
            logger.info(f"Running ablation: {ablation}")
            df = await run_geval_ablation(
                ablation=ablation,
                client=client,
                records=records,
                limit=evaluation_limit,
                score_mode=score_mode,
                n_samples=n_samples,
                results_dir=results_dir,
            )
            df["data_label"] = "MOCK DATA" if mock else "REAL DATA"
            suffix = "_mock" if mock else ""
            fname = f"geval_scores_{ablation}{suffix}.csv"
            save_results(df, results_dir / fname)
            frames.append(df)
        return frames

    all_dfs = asyncio.run(evaluate_ablations())

    all_scores = pd.concat(all_dfs, ignore_index=True)
    human_df = records_to_dataframe(records)
    if limit and not has_sampled_records:
        human_df = human_df.iloc[:limit]

    # Compute correlations
    corr_rows = []
    for ablation in ablation_list:
        abl_df = all_scores[all_scores["ablation"] == ablation]
        for dim in list(DIMENSIONS) + ["overall"]:
            dim_df = abl_df[abl_df["dimension"] == dim].copy()
            if len(dim_df) == 0:
                continue
            merged = dim_df.merge(
                human_df[["doc_id", "system_id"] + [f"human_{d}" for d in HUMAN_DIMENSIONS]],
                on=["doc_id", "system_id"],
                how="left",
            )
            pred = merged["weighted_score"].values.astype(float)
            for human_dim in HUMAN_DIMENSIONS:
                human = merged[f"human_{human_dim}"].values.astype(float)
                valid = ~(np.isnan(pred) | np.isnan(human))
                if valid.sum() < 10:
                    continue
                p, h = pred[valid], human[valid]
                d_ids = merged["doc_id"].values[valid]
                s_ids = merged["system_id"].values[valid]

                sum_corr = summary_level_correlation(p, h, d_ids)
                samp_corr = pooled_correlation(p, h)
                sys_corr = system_level_correlation(p, h, s_ids)
                bootstrap_cfg = cfg.get("bootstrap", {})
                _, ci_low, ci_high = bootstrap_summary_level_ci(
                    p,
                    h,
                    d_ids,
                    statistic="spearman",
                    n_resamples=bootstrap_cfg.get("n_resamples", 1000),
                    confidence=bootstrap_cfg.get("confidence_level", 0.95),
                    seed=seed,
                )

                corr_rows.append({
                    "ablation": ablation,
                    "geval_dim": dim,
                    "human_dim": human_dim,
                    "summary_spearman": sum_corr["spearman"],
                    "summary_spearman_ci_lo": ci_low,
                    "summary_spearman_ci_hi": ci_high,
                    "summary_kendall": sum_corr["kendall"],
                    "summary_pearson": sum_corr["pearson"],
                    "summary_n_docs": sum_corr["n_docs"],
                    "sample_spearman": samp_corr["spearman"],
                    "sample_kendall": samp_corr["kendall"],
                    "sample_pearson": samp_corr["pearson"],
                    "system_spearman": sys_corr["spearman"],
                    "system_kendall": sys_corr["kendall"],
                    "system_pearson": sys_corr["pearson"],
                })

    corr_df = pd.DataFrame(corr_rows)
    suffix = "_mock" if mock else ""
    corr_df["data_label"] = "MOCK DATA" if mock else "REAL DATA"
    save_results(corr_df, results_dir / f"exp_a_correlations{suffix}.csv")

    # Print comparison with paper
    print("\n=== Experiment A: Summary-Level Correlations (Full G-Eval) ===")
    full_corr = corr_df[
        (corr_df["ablation"] == "full")
        & (corr_df["geval_dim"] == corr_df["human_dim"])
        & (corr_df["geval_dim"].isin(list(DIMENSIONS)))
    ]
    if not full_corr.empty:
        print(full_corr[["geval_dim", "summary_spearman", "summary_kendall"]].to_string(index=False))
        avg_spearman = full_corr["summary_spearman"].mean()
        avg_kendall = full_corr["summary_kendall"].mean()
        print(f"\nOur avg Spearman: {avg_spearman:.4f} (paper: 0.514)")
        print(f"Our avg Kendall:  {avg_kendall:.4f} (paper: 0.418)")
        if mock:
            print("[NOTE: These are MOCK DATA results, not real G-Eval scores]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment A: G-Eval Baseline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--ablation", nargs="+", default=None, choices=list(ABLATION_MODES.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment_a(
        limit=args.limit,
        mock=args.mock,
        results_dir=args.results_dir,
        ablations=args.ablation,
        seed=args.seed,
    )
