"""Run a bounded, single-judge pilot suitable for a quick report draft."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_exp_a import run_experiment_a
from experiments.run_exp_b import run_experiment_b
from experiments.run_exp_e import run_experiment_e
from experiments.run_exp_f import run_experiment_f
from experiments.run_exp_g import run_experiment_g
from src.human_alignment import run_experiment_d
from src.llm_clients import check_api_key_available
from src.utils import load_config, load_summeval, select_quick_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pilot scope and estimated G-Eval calls without using an API.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run the same bounded pipeline with MockLLMClient instead of an API.",
    )
    args = parser.parse_args()

    cfg = load_config()
    quick_cfg = cfg.get("quick_run", {})
    records = select_quick_sample(
        load_summeval(),
        n_documents=int(quick_cfg.get("n_documents", 10)),
        n_systems_per_document=int(quick_cfg.get("n_systems_per_document", 2)),
    )
    provider = "mock" if args.mock else cfg.get("llm", {}).get("provider", "unknown")
    key_name = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider, "the configured provider's API key")
    calls_per_score = (
        int(cfg["geval"]["n_samples"])
        if cfg["geval"]["score_mode"] == "sampling"
        else 1
    )
    expected_calls = (
        len(records) * len(cfg["geval"]["dimensions"]) * calls_per_score
    )
    if args.dry_run:
        print(
            f"Pilot dry run: provider={provider}, "
            f"model={'mock-model' if args.mock else cfg['llm']['model']}, "
            f"records={len(records)}, documents={len({r['doc_id'] for r in records})}, "
            f"systems={sorted({r['system_id'] for r in records})}, "
            f"estimated G-Eval requests<={0 if args.mock else expected_calls} before retries."
        )
        return
    if not args.mock and not check_api_key_available(cfg):
        raise SystemExit(
            f"No API key is configured for provider '{provider}'. "
            f"Set {key_name} in .env, then rerun this command."
        )

    results_dir = (
        Path(cfg["mock_results_dir"]) / "fast"
        if args.mock
        else Path(cfg["results_dir"]) / "fast"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running pilot on %d records from %d documents and %d systems. "
        "Expected external G-Eval API calls: at most %d.",
        len(records),
        len({record["doc_id"] for record in records}),
        len({record["system_id"] for record in records}),
        0 if args.mock else expected_calls,
    )

    run_experiment_b(records=records, results_dir=results_dir, mock=args.mock)
    run_experiment_a(
        records=records,
        mock=args.mock,
        results_dir=results_dir,
        ablations=[quick_cfg.get("ablation", "no_cot")],
    )
    run_experiment_e(records=records, mock=args.mock, results_dir=results_dir)
    run_experiment_f(records=records, mock=args.mock, results_dir=results_dir)
    run_experiment_g(
        records=records,
        mock=args.mock,
        results_dir=results_dir,
        weights={
            "geval": 0.50,
            "bertscore": 0.30,
            "rouge": 0.20,
            "ensemble_agreement": 0.0,
        },
    )
    human_template = run_experiment_d(
        records=load_summeval(),
        n_samples=int(quick_cfg.get("human_rating_samples", 160)),
        results_dir=results_dir,
    )

    metadata = {
        "mode": "mock pilot" if args.mock else "single-judge pilot",
        "provider": provider,
        "model": "mock-model" if args.mock else cfg["llm"]["model"],
        "is_mock": args.mock,
        "records": len(records),
        "documents": len({record["doc_id"] for record in records}),
        "systems": sorted({record["system_id"] for record in records}),
        "dimensions": cfg["geval"]["dimensions"],
        "ablation": quick_cfg.get("ablation", "no_cot"),
        "expected_geval_api_calls_max": (
            0 if args.mock else len(records) * len(cfg["geval"]["dimensions"])
        ),
        "ensemble_comparison": "omitted; only one judge is used",
        "human_rating_template": human_template["template_path"],
        "limitations": [
            "Small deterministic subset; correlations and confidence intervals are exploratory.",
            "No multi-judge ensemble comparison or fresh summary generation.",
            "Manual human ratings are not included until collected.",
        ],
    }
    with (results_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    logger.info("Pilot outputs saved under %s", results_dir)


if __name__ == "__main__":
    main()
