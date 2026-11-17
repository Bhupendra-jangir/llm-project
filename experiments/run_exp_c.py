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
    ensure_results_dir, save_results, DIMENSIONS, summary_level_correlation,
    system_level_correlation,
)
from src.llm_clients import create_client, DiskCache, check_api_key_available
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
                           concurrency_limit=cfg.get("concurrency_limit", 5),
                           max_retries=cfg.get("max_retries", 5),
                           retry_base_delay=cfg.get("retry_base_delay", 1.0),
                           requests_per_minute=cfg.get("llm", {}).get("requests_per_minute", 60))
    evaluator = GEvalEvaluator(
        client=client, dimensions=list(DIMENSIONS),
        score_mode=cfg["geval"]["score_mode"],
        n_samples=cfg["geval"]["n_samples"],
    )
    estimate = evaluator.estimate_cost(
        min(limit or len(records), len(records)),
        avg_prompt_tokens=cfg.get("cost_estimate", {}).get(
            "average_input_tokens_per_record", 500
        ),
        avg_output_tokens=cfg.get("cost_estimate", {}).get(
            "average_output_tokens_per_record", 5
        ),
    )
    logger.info("Judge %s cost estimate: %s", label, estimate)
    df = await evaluator.evaluate_dataset(records, limit=limit, show_progress=True)
    df["judge"] = label
    df["data_label"] = "MOCK DATA" if is_mock else "REAL DATA"
    return df


def run_experiment_c(
    limit: int | None = None,
    mock: bool = False,
    results_dir: Path | None = None,
    seed: int = 42,
) -> None:
    cfg = load_config()
    set_random_seed(seed)
    if not mock and (
        cfg.get("llm", {}).get("provider", "mock") == "mock"
        or not check_api_key_available(cfg)
    ):
        print(
            "No configured real judge key was found. Experiment C will use "
            "MockLLMClient; set .env and configs/config.yaml for real judges."
        )
        mock = True
    records = load_summeval()
    human_df = records_to_dataframe(records)

    judge_configs = cfg.get("judge_models", [])
    if mock:
        judge_configs = [
            {"provider": "mock", "model": "mock-a", "label": "Mock-Judge-A"},
            {"provider": "mock", "model": "mock-b", "label": "Mock-Judge-B"},
            {"provider": "mock", "model": "mock-c", "label": "Mock-Judge-C"},
        ]
    else:
        if not judge_configs:
            raise ValueError("Configure at least two real judges in judge_models for Experiment C")
        if any(judge["provider"] == "mock" for judge in judge_configs):
            raise ValueError(
                "Experiment C cannot mix real and mock judges. Configure at least "
                "two real judge models or run with --mock."
            )
        missing_keys = [
            judge["provider"]
            for judge in judge_configs
            if not check_api_key_available(cfg, provider=judge["provider"])
        ]
        if missing_keys:
            providers = ", ".join(sorted(set(missing_keys)))
            raise RuntimeError(
                f"Missing API key(s) for configured judge provider(s): {providers}. "
                "Set their keys in .env or run with --mock."
            )
    res_dir = (
        Path(cfg["mock_results_dir"])
        if mock
        else results_dir or Path(cfg["results_dir"])
    )
    ensure_results_dir(res_dir)

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
                              "summary_kendall": sc["kendall"],
                              "data_label": "MOCK DATA" if mock else "REAL DATA"})

    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        save_results(corr_df, res_dir / f"exp_c_correlations{suffix}.csv")
        print("\n=== Experiment C: Judge Correlations (summary-level Spearman) ===")
        try:
            print(corr_df.pivot_table(index="judge", columns="dimension",
                                      values="summary_spearman").round(4).to_string())
        except Exception:
            print(corr_df.to_string())

    ranking_rows = []
    for jcfg in judge_configs:
        label = jcfg.get("label", f"{jcfg['provider']}/{jcfg['model']}")
        judge_overall = combined[
            (combined["judge"] == label) & (combined["dimension"] == "overall")
        ].merge(
            human_df[["doc_id", "system_id", "human_overall"]],
            on=["doc_id", "system_id"],
            how="inner",
            validate="one_to_one",
        )
        by_system = judge_overall.groupby("system_id", as_index=False).agg(
            judge_score=("weighted_score", "mean"),
            human_score=("human_overall", "mean"),
        )
        tau = system_level_correlation(
            by_system["judge_score"].to_numpy(),
            by_system["human_score"].to_numpy(),
            by_system["system_id"].to_numpy(),
        )["kendall"]
        by_system["judge"] = label
        by_system["judge_rank"] = by_system["judge_score"].rank(ascending=False, method="min")
        by_system["human_rank"] = by_system["human_score"].rank(ascending=False, method="min")
        by_system["kendall_tau_vs_human"] = tau
        by_system["data_label"] = "MOCK DATA" if mock else "REAL DATA"
        ranking_rows.extend(by_system.to_dict(orient="records"))
    ranking_df = pd.DataFrame(ranking_rows)
    if not ranking_df.empty:
        save_results(ranking_df, res_dir / f"exp_c_system_rankings{suffix}.csv")

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
