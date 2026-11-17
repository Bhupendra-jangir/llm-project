"""Experiment D: human validation subset and alignment with expert scores."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.human_alignment import run_experiment_d
from src.utils import DIMENSIONS, load_config, load_summeval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_automatic_metrics(results_dir: Path) -> pd.DataFrame | None:
    """Combine saved traditional and G-Eval metrics when available."""
    metrics_path = results_dir / "traditional_metrics.csv"
    if not metrics_path.exists():
        logger.info("Traditional metrics are missing at %s", metrics_path)
        return None

    metrics = pd.read_csv(metrics_path)
    geval_path = results_dir / "geval_scores_full_mock.csv"
    if not geval_path.exists():
        geval_path = results_dir / "geval_scores_full.csv"
    if geval_path.exists():
        geval = pd.read_csv(geval_path)
        for dimension in [*DIMENSIONS, "overall"]:
            dimension_scores = geval.loc[
                geval["dimension"] == dimension,
                ["doc_id", "system_id", "weighted_score"],
            ].rename(columns={"weighted_score": f"geval_{dimension}"})
            metrics = metrics.merge(
                dimension_scores,
                on=["doc_id", "system_id"],
                how="left",
                validate="one_to_one",
            )

    judge_path = results_dir / "exp_c_judge_scores_mock.csv"
    if not judge_path.exists():
        judge_path = results_dir / "exp_c_judge_scores.csv"
    if judge_path.exists():
        judges = pd.read_csv(judge_path)
        overall = judges.loc[judges["dimension"] == "overall"]
        agreement = (
            overall.groupby(["doc_id", "system_id"])["weighted_score"]
            .std()
            .rename("geval_judge_std")
            .reset_index()
        )
        metrics = metrics.merge(
            agreement,
            on=["doc_id", "system_id"],
            how="left",
            validate="one_to_one",
        )
    return metrics


def run_human_alignment(
    n_samples: int | None = None,
    seed: int = 42,
    mock: bool = False,
    results_dir: Path | None = None,
) -> dict:
    """Create the human rating sheet and compare available metrics to experts."""
    cfg = load_config()
    if results_dir is None:
        results_dir = Path(
            cfg["mock_results_dir"] if mock else cfg["results_dir"]
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    sample_count = (
        cfg.get("human_validation", {}).get("n_samples", 160)
        if n_samples is None
        else n_samples
    )
    ratings_path = Path(cfg["data"]["human_ratings_path"])
    if not ratings_path.is_absolute():
        ratings_path = ROOT / ratings_path

    records = load_summeval()
    automatic_metrics = load_automatic_metrics(results_dir)
    output = run_experiment_d(
        records=records,
        n_samples=sample_count,
        seed=seed,
        results_dir=results_dir,
        human_ratings_path=ratings_path,
        automatic_metrics=automatic_metrics,
        rating_sheet_path=results_dir / "human_rating_sheet.csv",
        data_label="MOCK DATA" if mock else "REAL DATA",
    )
    print("\n=== Experiment D: Human Alignment ===")
    print(f"Subset size: {output['subset_size']}")
    print(f"Rating sheet: {output['template_path']}")
    print(output["human_reference"])
    if automatic_metrics is None:
        print("No automatic metric results were available for comparison.")
    if output["inter_rater_agreement"] is None:
        print("No new human ratings were imported; no ratings were fabricated.")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment D: Human Validation")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    run_human_alignment(
        n_samples=args.n_samples,
        seed=args.seed,
        mock=args.mock,
        results_dir=args.results_dir,
    )
