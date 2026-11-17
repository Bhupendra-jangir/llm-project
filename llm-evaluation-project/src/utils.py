"""
src/utils.py
============
Shared utilities: data loading, caching, statistics, bootstrap CI.
All functions are pure (no side-effects on shared state) and type-annotated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MOCK_RESULTS_DIR = RESULTS_DIR / "_mock"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
HUMAN_DIMENSIONS = ["coherence", "consistency", "fluency", "relevance", "overall"]
SCORE_RANGES = {
    "coherence": (1, 5),
    "consistency": (1, 5),
    "fluency": (1, 3),
    "relevance": (1, 5),
}
SYSTEM_IDS = [f"M{i}" for i in range(24)]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and return the YAML configuration."""
    path = config_path or CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_summeval(
    path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load the SummEval dataset from JSON.

    Args:
        path: Override path to summeval.json.
        limit: If set, return only the first N records.

    Returns:
        List of record dicts with keys: doc_id, system_id, source, reference,
        system_output, scores.
    """
    fpath = path or DATA_DIR / "summeval.json"
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return data


def validate_summeval(data: List[Dict[str, Any]]) -> None:
    """Validate schema and raise on errors."""
    required_keys = {"doc_id", "system_id", "source", "reference", "system_output", "scores"}
    score_keys = set(HUMAN_DIMENSIONS)
    for i, rec in enumerate(data):
        missing = required_keys - set(rec.keys())
        if missing:
            raise ValueError(f"Record {i} missing keys: {missing}")
        missing_scores = score_keys - set(rec["scores"].keys())
        if missing_scores:
            raise ValueError(f"Record {i} missing score keys: {missing_scores}")


def print_dataset_stats(data: List[Dict[str, Any]]) -> None:
    """Print dataset statistics to stdout."""
    df = records_to_dataframe(data)
    n_records = len(df)
    n_docs = df["doc_id"].nunique()
    n_systems = df["system_id"].nunique()
    system_ids = sorted(df["system_id"].unique())

    print(f"\n{'='*60}")
    print(f"SummEval Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total records   : {n_records}")
    print(f"Unique documents: {n_docs}")
    print(f"Unique systems  : {n_systems}")
    print(f"System IDs      : {system_ids}")
    print(f"\nHuman score distributions (mean ± std):")
    for dim in HUMAN_DIMENSIONS:
        col = f"human_{dim}"
        print(f"  {dim:12s}: {df[col].mean():.3f} ± {df[col].std():.3f} "
              f"[{df[col].min():.2f}, {df[col].max():.2f}]")
    print(f"{'='*60}\n")


def records_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of SummEval records to a flat DataFrame."""
    rows = []
    for rec in data:
        row = {
            "doc_id": rec["doc_id"],
            "system_id": rec["system_id"],
            "source": rec["source"],
            "reference": rec["reference"],
            "system_output": rec["system_output"],
        }
        for dim in HUMAN_DIMENSIONS:
            row[f"human_{dim}"] = rec["scores"].get(dim, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Disk Cache (SQLite)
# ---------------------------------------------------------------------------

class DiskCache:
    """SQLite-backed disk cache keyed by SHA-256 hash of request params."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL
                )"""
            )
            conn.commit()

    @staticmethod
    def make_key(**kwargs: Any) -> str:
        """Create a deterministic cache key from keyword arguments."""
        canonical = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )
            conn.commit()

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


# ---------------------------------------------------------------------------
# Score Parsing
# ---------------------------------------------------------------------------

def parse_score(text: str, min_score: float = 1.0, max_score: float = 5.0) -> Optional[float]:
    """Extract a numeric score from LLM output text.

    Strategy:
      1. Try to find a number at the start (possibly after a dash/colon).
      2. Fall back to any float/int in the string.
      3. Return None if nothing parseable.
    """
    if not text:
        return None
    text = text.strip()

    # Pattern 1: leading number, possibly preceded by "- Dimension:"
    m = re.search(r"[-:\s]*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        try:
            val = float(m.group(1))
            if min_score <= val <= max_score:
                return val
        except ValueError:
            pass

    # Pattern 2: any float in the string
    nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", text)
    for n in nums:
        try:
            val = float(n)
            if min_score <= val <= max_score:
                return val
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# Correlation Utilities
# ---------------------------------------------------------------------------

def summary_level_correlation(
    pred: np.ndarray,
    human: np.ndarray,
    doc_ids: np.ndarray,
) -> Dict[str, float]:
    """Compute per-document correlations, then average (SummEval protocol).

    Args:
        pred:    Predicted scores, shape (N,).
        human:   Human scores, shape (N,).
        doc_ids: Document ID per record, shape (N,).

    Returns:
        Dict with 'spearman', 'kendall', 'pearson' averaged over documents.
    """
    unique_docs = np.unique(doc_ids)
    spearman_vals, kendall_vals, pearson_vals = [], [], []

    for doc in unique_docs:
        mask = doc_ids == doc
        p = pred[mask]
        h = human[mask]
        # Skip if constant (correlation undefined)
        if len(p) < 2 or np.std(p) == 0 or np.std(h) == 0:
            continue
        spearman_vals.append(stats.spearmanr(p, h).statistic)
        kendall_vals.append(stats.kendalltau(p, h).statistic)
        pearson_vals.append(stats.pearsonr(p, h).statistic)

    return {
        "spearman": float(np.mean(spearman_vals)) if spearman_vals else float("nan"),
        "kendall": float(np.mean(kendall_vals)) if kendall_vals else float("nan"),
        "pearson": float(np.mean(pearson_vals)) if pearson_vals else float("nan"),
        "n_docs": len(spearman_vals),
    }


def pooled_correlation(pred: np.ndarray, human: np.ndarray) -> Dict[str, float]:
    """Sample-level (pooled) correlations."""
    if len(pred) < 2 or np.std(pred) == 0 or np.std(human) == 0:
        return {"spearman": float("nan"), "kendall": float("nan"), "pearson": float("nan")}
    return {
        "spearman": float(stats.spearmanr(pred, human).statistic),
        "kendall": float(stats.kendalltau(pred, human).statistic),
        "pearson": float(stats.pearsonr(pred, human).statistic),
    }


def system_level_correlation(
    pred: np.ndarray,
    human: np.ndarray,
    system_ids: np.ndarray,
) -> Dict[str, float]:
    """Aggregate per-system means, then correlate."""
    unique_sys = np.unique(system_ids)
    pred_means = np.array([pred[system_ids == s].mean() for s in unique_sys])
    human_means = np.array([human[system_ids == s].mean() for s in unique_sys])
    return pooled_correlation(pred_means, human_means)


def top_k_overlap(
    pred: np.ndarray,
    human: np.ndarray,
    system_ids: np.ndarray,
    k: int = 5,
) -> float:
    """Compute overlap in top-k systems between metric and human rankings."""
    unique_sys = np.unique(system_ids)
    pred_means = {s: pred[system_ids == s].mean() for s in unique_sys}
    human_means = {s: human[system_ids == s].mean() for s in unique_sys}
    top_pred = set(sorted(pred_means, key=pred_means.get, reverse=True)[:k])
    top_human = set(sorted(human_means, key=human_means.get, reverse=True)[:k])
    return len(top_pred & top_human) / k


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    pred: np.ndarray,
    human: np.ndarray,
    doc_ids: np.ndarray,
    stat_fn: Callable,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI by resampling documents (not individual samples).

    Args:
        pred:       Predicted scores.
        human:      Human scores.
        doc_ids:    Document ID per record.
        stat_fn:    Function(pred, human, doc_ids) -> scalar.
        n_resamples: Number of bootstrap iterations.
        confidence: CI level (e.g. 0.95).
        seed:       Random seed.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    unique_docs = np.unique(doc_ids)
    n_docs = len(unique_docs)

    point = stat_fn(pred, human, doc_ids)
    boot_stats = []
    for _ in range(n_resamples):
        sampled_docs = rng.choice(unique_docs, size=n_docs, replace=True)
        # Build bootstrap sample
        indices = np.concatenate([np.where(doc_ids == d)[0] for d in sampled_docs])
        b_pred = pred[indices]
        b_human = human[indices]
        b_docs = doc_ids[indices]
        try:
            boot_stats.append(stat_fn(b_pred, b_human, b_docs))
        except Exception:
            pass

    if not boot_stats:
        return point, float("nan"), float("nan")

    alpha = 1 - confidence
    lower = float(np.nanpercentile(boot_stats, 100 * alpha / 2))
    upper = float(np.nanpercentile(boot_stats, 100 * (1 - alpha / 2)))
    return float(point), lower, upper


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def z_score_normalize(arr: np.ndarray) -> np.ndarray:
    """Z-score normalise an array, handling zero std."""
    std = arr.std()
    if std == 0:
        return arr - arr.mean()
    return (arr - arr.mean()) / std


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    rng = arr.max() - arr.min()
    if rng == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / rng


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def ensure_results_dir(path: Path | str) -> Path:
    """Create the results directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_results(df: pd.DataFrame, path: Path | str) -> None:
    """Save a DataFrame to CSV, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logger.info(f"Saved results to {p}")


def load_results(path: Path | str) -> pd.DataFrame:
    """Load a CSV results file."""
    return pd.read_csv(Path(path))


# ---------------------------------------------------------------------------
# Token / Cost Logging
# ---------------------------------------------------------------------------

class CostTracker:
    """Track token usage and estimated cost across API calls."""

    # Approximate cost per 1k tokens (USD) — update as pricing changes
    COST_TABLE = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "default": {"input": 0.001, "output": 0.003},
    }

    def __init__(self, model: str = "default") -> None:
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def log(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1

    @property
    def estimated_cost_usd(self) -> float:
        rates = self.COST_TABLE.get(self.model, self.COST_TABLE["default"])
        return (
            self.input_tokens / 1000 * rates["input"]
            + self.output_tokens / 1000 * rates["output"]
        )

    def report(self) -> str:
        return (
            f"API calls: {self.calls}, "
            f"Input tokens: {self.input_tokens:,}, "
            f"Output tokens: {self.output_tokens:,}, "
            f"Est. cost: ${self.estimated_cost_usd:.4f}"
        )


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)


def word_count(text: str) -> int:
    """Count whitespace-separated tokens."""
    return len(text.split())


def compute_extractive_coverage(summary: str, source: str) -> float:
    """Fraction of unigrams in the summary covered by the source."""
    summary_tokens = set(summary.lower().split())
    source_tokens = set(source.lower().split())
    if not summary_tokens:
        return 0.0
    return len(summary_tokens & source_tokens) / len(summary_tokens)


def compute_extractive_density(summary: str, source: str) -> float:
    """Average length of extractive fragments (Grusky et al.)."""
    summary_words = summary.lower().split()
    source_words = source.lower().split()
    if not summary_words:
        return 0.0
    total_len = 0
    i = 0
    while i < len(summary_words):
        j = i
        # Greedily extend the matching fragment
        frag_len = 0
        best = 0
        for k in range(len(source_words)):
            length = 0
            while (i + length < len(summary_words) and
                   k + length < len(source_words) and
                   summary_words[i + length] == source_words[k + length]):
                length += 1
            best = max(best, length)
        total_len += best ** 2
        i += max(best, 1)
    return total_len / len(summary_words)
