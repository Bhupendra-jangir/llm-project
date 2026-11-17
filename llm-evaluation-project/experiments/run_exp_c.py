"""
experiments/run_exp_c.py
Experiment C: Multi-model judge comparison.
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
    summary_level_correlation,
)
from src.llm_clients import create_client, DiskCache
from src.geval_evaluator import GEvalEvaluator
from src.generate_responses import run_generation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run_judge(judge_cfg: dict, records: list, limit: int | None, cfg: dict, is_mock: bool) -> pd.DataFrame:
    provider = judge_cfg["provider"]
    model = judge_cfg["model"]
    label = judge_cfg.get("label", f"{provider}/{model}")
    cache = DiskCache(cfg["cache_db"]) if not is_mock else None
    client = create_client(provider=provider, model=model, cache=cache,
                           concurrency_limit=cfg.get("concurrency_limit", 5))
    evaluator = GEvalEvaluator(
        client=client, dimensions=list(DIMENSIONS),
        score_mode=cfg["geval"]["score_mode"],
        n_samples=cfg["geval"]["n_samples"],
    )
    df = await evaluator.evaluate_dataset(records, limit=limit, show_progress=True)
    df["judge"] = label
    return df


def run_experiment_c(
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
    human_df = records_to_dataframe(records)

    judge_configs = cfg.get("judge_models", [])
    if mock or not judge_configs:
        judge_configs = [
            {"provider": "mock", "model": "mock-a", "label": "Mock-Judge-A"},
            {"provider": "mock", "model": "mock-b", "label": "Mock-Judge-B"},
            {"provider": "mock", "model": "mock-c", "label": "Mock-Judge-C"},
        ]

    logger.info(f"=== Experiment C: Multi-Model Comparison ({len(judge_configs)} judges) ===")
    all_judge_dfs = []
    for jcfg in judge_configs:
        df = asyncio.run(run_judge(jcfg, records, limit, cfg, mock))
        all_judge_dfs.append(df)

    combined = pd.concat(all_judge_dfs, ignore_index=True)
    suffix = "_mock" if mock else ""
    save_results(combined, res_dir / f"exp_c_judge_scores{suffix}.csv")

    # Correlations per judge vs human
    corr_rows = []
    for dim in DIMENSIONS:
        for jcfg in judge_configs:
            label = jcfg.get("label", "")
            jdf = combined[(combined["judge"] == label) & (combined["dimension"] == dim)].copy()
            merged = jdf.merge(
                human_df[["doc_id", "system_id", f"human_{dim}"]],
                on=["doc_id", "system_id"], how="left"
            )
            pred = merged["weighted_score"].values.astype(float)
            human = merged[f"human_{dim}"].values.astype(float)
            valid = ~(np.isnan(pred) | np.isnan(human))
            if valid.sum() < 10:
                continue
            sc = summary_level_correlation(
                pred[valid], human[valid], merged["doc_id"].values[valid]
            )
            corr_rows.append({"judge": label, "dimension": dim,
                              "summary_spearman": sc["spearman"],
                              "summary_kendall": sc["kendall"]})

    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        save_results(corr_df, res_dir / f"exp_c_correlations{suffix}.csv")
        print("\n=== Experiment C: Judge Correlations (summary-level Spearman) ===")
        try:
            print(corr_df.pivot_table(index="judge", columns="dimension",
                                      values="summary_spearman").round(4).to_string())
        except Exception:
            print(corr_df.to_string())

    # Optional fresh generation
    gen_df = run_generation(cfg)
    if gen_df is not None:
        logger.info(f"Fresh summaries generated: {len(gen_df)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment C: Multi-Model Comparison")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    run_experiment_c(limit=args.limit, mock=args.mock, results_dir=args.results_dir)
