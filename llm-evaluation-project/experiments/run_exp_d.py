"""
experiments/run_exp_d.py
Experiment D: Human validation subset.
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_summeval, load_config, set_random_seed
from src.human_alignment import run_experiment_d

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment D: Human Validation")
    parser.add_argument("--n-samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_random_seed(args.seed)
    records = load_summeval()
    res_dir = args.results_dir or Path(cfg["results_dir"])

    result = run_experiment_d(
        records=records,
        n_samples=args.n_samples,
        seed=args.seed,
        results_dir=res_dir,
    )
    print(f"\n=== Experiment D Results ===")
    print(f"Subset size: {result['subset_size']}")
    print(f"Template exported to: {result['template_path']}")
    if result["inter_rater_agreement"] is None:
        print("No human ratings found. Add ratings to data/human_ratings.csv to compute IRR.")
    else:
        print(f"Inter-rater agreement (n={result['inter_rater_agreement']['n_records']}):")
        for dim, vals in result["inter_rater_agreement"]["dimensions"].items():
            print(f"  {dim}: Spearman={vals.get('spearman', 'N/A'):.3f}")
