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
from src.human_alignment import run_experiment_d
from src.utils import load_summeval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    kw = dict(limit=args.limit, mock=args.mock, seed=args.seed)

    logger.info("Phase 1: Traditional metrics (Exp B)")
    run_experiment_b(limit=args.limit, seed=args.seed)

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
    records = load_summeval()
    run_experiment_d(records=records, seed=args.seed)

    logger.info("All experiments complete!")
