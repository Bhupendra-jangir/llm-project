"""
experiments/run_all.py
Master runner: runs all experiments in order.
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_exp_b import run_experiment_b
from experiments.run_exp_a import run_experiment_a
from experiments.run_exp_c import run_experiment_c
from experiments.run_exp_e import run_experiment_e
from experiments.run_exp_f import run_experiment_f
from experiments.run_exp_g import run_experiment_g
from experiments.run_exp_d import run_human_alignment
from src.llm_clients import check_api_key_available
from src.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    llm_provider = cfg.get("llm", {}).get("provider", "mock")
    mock = (
        args.mock
        or llm_provider == "mock"
        or not check_api_key_available(cfg)
    )
    if not mock:
        judges = cfg.get("judge_models", [])
        if len(judges) < 2 or all(judge.get("provider") == "mock" for judge in judges):
            raise SystemExit(
                "Real full-pipeline runs require at least two real entries in "
                "configs/config.yaml: judge_models."
            )
        missing_judges = sorted(
            {
                judge["provider"]
                for judge in judges
                if not check_api_key_available(cfg, provider=judge["provider"])
            }
        )
        if missing_judges:
            raise SystemExit(
                "Missing API keys for configured judges: "
                + ", ".join(missing_judges)
                + ". Add keys to .env or run with --mock."
            )
    real_results_dir = (
        Path(cfg["results_dir"]) / "_pilot"
        if args.limit is not None and not mock
        else None
    )
    kw = dict(
        limit=args.limit,
        mock=mock,
        seed=args.seed,
        results_dir=real_results_dir,
    )

    logger.info("Phase 1: Traditional metrics (Exp B)")
    run_experiment_b(
        limit=args.limit,
        results_dir=Path(cfg["mock_results_dir"]) if mock else real_results_dir,
        seed=args.seed,
    )

    logger.info("Phase 2: G-Eval baseline (Exp A)")
    run_experiment_a(**kw)

    logger.info("Phase 3a: Multi-model comparison (Exp C)")
    run_experiment_c(**kw)

    logger.info("Phase 3b: Correlation matrix (Exp E)")
    run_experiment_e(**kw)

    logger.info("Phase 4a: Bias analysis (Exp F)")
    run_experiment_f(**kw)

    logger.info("Phase 4b: Unified score (Exp G)")
    run_experiment_g(**kw)

    logger.info("Phase 4c: Human validation (Exp D)")
    run_human_alignment(
        seed=args.seed,
        mock=mock,
        results_dir=real_results_dir,
    )

    logger.info("All experiments complete!")
