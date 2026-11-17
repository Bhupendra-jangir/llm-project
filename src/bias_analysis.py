"""
src/bias_analysis.py
====================
Evaluator bias analysis (Experiment F):

1. Per-system bias: mean z-scored (G-Eval - human), significance tests
2. Length bias: correlation of each metric with summary word count
3. Extractiveness/style bias: extractive coverage/density vs residuals
4. Self-preference: system type labels + (optional) judge-vs-own output
5. Position/verbosity sanity checks

Outputs plots (PNG, 300 dpi) and a summary CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.utils import (
    ensure_results_dir,
    z_score_normalize,
    word_count,
    compute_extractive_coverage,
    compute_extractive_density,
)

logger = logging.getLogger(__name__)

# System type labels (known from SummEval paper / literature)
SYSTEM_TYPE_LABELS: Dict[str, str] = {
    "M0": "extractive",
    "M1": "extractive",
    "M2": "extractive",
    "M5": "extractive",
    "M8": "neural_abstractive",
    "M9": "neural_abstractive",
    "M10": "neural_abstractive",
    "M11": "neural_abstractive",
    "M12": "neural_abstractive",
    "M13": "neural_abstractive",
    "M14": "neural_abstractive",
    "M15": "neural_abstractive",
    "M17": "neural_abstractive",
    "M20": "neural_abstractive",
    "M22": "neural_abstractive",
    "M23": "neural_abstractive",
}


# ---------------------------------------------------------------------------
# 1. Per-system bias
# ---------------------------------------------------------------------------

def compute_system_bias(
    df: pd.DataFrame,
    metric_col: str,
    human_col: str,
) -> pd.DataFrame:
    """Compute per-system bias = mean z-scored (metric - human).

    Args:
        df:         DataFrame with metric_col, human_col, system_id columns.
        metric_col: Column name of the metric score.
        human_col:  Column name of the human score.

    Returns:
        DataFrame with system_id, bias, std_error, t_stat, p_value.
    """
    sub = df[[metric_col, human_col, "system_id"]].dropna()
    z_metric = z_score_normalize(sub[metric_col].values)
    z_human = z_score_normalize(sub[human_col].values)
    residuals = z_metric - z_human

    rows = []
    for sys_id in sorted(sub["system_id"].unique()):
        mask = sub["system_id"] == sys_id
        res = residuals[mask.values]
        bias = float(np.mean(res))
        se = float(np.std(res, ddof=1) / np.sqrt(len(res))) if len(res) > 1 else float("nan")
        t_stat, p_val = stats.ttest_1samp(res, 0) if len(res) > 1 else (float("nan"), float("nan"))
        rows.append({
            "system_id": sys_id,
            "bias": bias,
            "std_error": se,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "overrated": bias > 0,
            "n": int(np.sum(mask)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Length bias
# ---------------------------------------------------------------------------

def compute_length_bias(
    df: pd.DataFrame,
    metric_col: str,
    human_col: str,
    text_col: str = "system_output",
) -> Dict[str, float]:
    """Correlation of metric with word count; partial correlation controlling for human score."""
    sub = df[[metric_col, human_col, text_col]].dropna()
    lengths = sub[text_col].apply(word_count).values
    metric = sub[metric_col].values
    human = sub[human_col].values

    rho_metric_len, _ = stats.spearmanr(metric, lengths)
    rho_human_len, _ = stats.spearmanr(human, lengths)

    # Partial correlation: metric ~ length | human score
    # Use linear regression residuals method
    try:
        import pingouin as pg
        partial_df = pd.DataFrame({"metric": metric, "length": lengths, "human": human})
        pc = pg.partial_corr(data=partial_df, x="metric", y="length", covar="human", method="spearman")
        partial_rho = float(pc["r"].iloc[0])
        partial_p = float(pc["p-val"].iloc[0])
    except ImportError:
        # Fallback: subtract regression component
        coeffs = np.polyfit(human, metric, 1)
        metric_resid = metric - np.polyval(coeffs, human)
        partial_rho, partial_p = stats.spearmanr(metric_resid, lengths)
        partial_rho = float(partial_rho)
        partial_p = float(partial_p)

    return {
        "spearman_metric_length": float(rho_metric_len),
        "spearman_human_length": float(rho_human_len),
        "partial_spearman_metric_length_given_human": partial_rho,
        "partial_p_value": partial_p,
    }


# ---------------------------------------------------------------------------
# 3. Extractiveness / style bias
# ---------------------------------------------------------------------------

def compute_extractiveness_bias(
    df: pd.DataFrame,
    metric_col: str,
    human_col: str,
) -> Dict[str, float]:
    """Test metric-vs-human residual against extractive coverage/density."""
    sub = df[[metric_col, human_col, "system_output", "source"]].dropna()
    coverage = sub.apply(
        lambda r: compute_extractive_coverage(r["system_output"], r["source"]), axis=1
    ).values
    density = sub.apply(
        lambda r: compute_extractive_density(r["system_output"], r["source"]), axis=1
    ).values

    # Residual: metric after removing human effect
    coeffs = np.polyfit(sub[human_col].values, sub[metric_col].values, 1)
    residual = sub[metric_col].values - np.polyval(coeffs, sub[human_col].values)

    rho_cov, p_cov = stats.spearmanr(residual, coverage)
    rho_den, p_den = stats.spearmanr(residual, density)

    return {
        "residual_vs_coverage_rho": float(rho_cov),
        "residual_vs_coverage_p": float(p_cov),
        "residual_vs_density_rho": float(rho_den),
        "residual_vs_density_p": float(p_den),
    }


# ---------------------------------------------------------------------------
# 4. System-type self-preference
# ---------------------------------------------------------------------------

def compute_system_type_bias(
    df: pd.DataFrame,
    metric_col: str,
    human_col: str,
) -> pd.DataFrame:
    """Group bias by system type (extractive vs neural_abstractive)."""
    sub = df[[metric_col, human_col, "system_id"]].dropna().copy()
    sub["system_type"] = sub["system_id"].map(SYSTEM_TYPE_LABELS).fillna("unknown")
    z_metric = z_score_normalize(sub[metric_col].values)
    z_human = z_score_normalize(sub[human_col].values)
    sub["residual"] = z_metric - z_human

    rows = []
    for stype in sorted(sub["system_type"].unique()):
        mask = sub["system_type"] == stype
        res = sub.loc[mask, "residual"].values
        rows.append({
            "system_type": stype,
            "mean_bias": float(np.mean(res)),
            "std": float(np.std(res, ddof=1)) if len(res) > 1 else float("nan"),
            "n": int(len(res)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_system_bias(
    bias_df: pd.DataFrame,
    metric_name: str,
    output_path: Path,
    is_mock: bool = False,
) -> None:
    """Plot per-system bias bars."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#e74c3c" if b > 0 else "#3498db" for b in bias_df["bias"]]
    ax.bar(bias_df["system_id"], bias_df["bias"], color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("System ID", fontsize=12)
    ax.set_ylabel("Mean Z-score Bias (metric − human)", fontsize=12)
    title = f"Per-System Bias: {metric_name}"
    if is_mock:
        title += " [MOCK DATA]"
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", rotation=45)

    # Add significance markers
    for i, (_, row) in enumerate(bias_df.iterrows()):
        if not pd.isna(row["p_value"]) and row["p_value"] < 0.05:
            ax.text(i, row["bias"] + 0.01 * np.sign(row["bias"]), "*", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved system bias plot to {output_path}")


def plot_residual_distribution(
    df: pd.DataFrame,
    metric_col: str,
    human_col: str,
    output_path: Path,
    is_mock: bool = False,
) -> None:
    """Plot residual distribution (metric z-score minus human z-score)."""
    sub = df[[metric_col, human_col]].dropna()
    z_metric = z_score_normalize(sub[metric_col].values)
    z_human = z_score_normalize(sub[human_col].values)
    residuals = z_metric - z_human

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40, color="#2ecc71", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero bias")
    ax.axvline(np.mean(residuals), color="orange", linestyle="-",
               linewidth=1.5, label=f"Mean={np.mean(residuals):.3f}")
    ax.set_xlabel("Residual (z-metric − z-human)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    title = f"Residual Distribution: {metric_col}"
    if is_mock:
        title += " [MOCK DATA]"
    ax.set_title(title, fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Full Experiment F runner
# ---------------------------------------------------------------------------

def run_bias_analysis(
    df: pd.DataFrame,
    metric_col: str,
    human_col: str,
    metric_name: str,
    results_dir: Path,
    is_mock: bool = False,
) -> Dict[str, Any]:
    """Run all five bias analyses for one metric-human dimension pair."""
    ensure_results_dir(results_dir)
    ensure_results_dir(results_dir / "figures")

    results: Dict[str, Any] = {
        "metric": metric_name,
        "human_dim": human_col,
    }

    # 1. Per-system bias
    sys_bias = compute_system_bias(df, metric_col, human_col)
    results["system_bias"] = sys_bias.to_dict(orient="records")
    plot_system_bias(
        sys_bias, metric_name,
        results_dir / "figures" / f"system_bias_{metric_name}_{human_col}.png",
        is_mock=is_mock,
    )

    # 2. Length bias
    if "system_output" in df.columns:
        length_bias = compute_length_bias(df, metric_col, human_col)
        results["length_bias"] = length_bias

    # 3. Extractiveness bias
    if "system_output" in df.columns and "source" in df.columns:
        ext_bias = compute_extractiveness_bias(df, metric_col, human_col)
        results["extractiveness_bias"] = ext_bias

    # 4. System type bias
    stype_bias = compute_system_type_bias(df, metric_col, human_col)
    results["system_type_bias"] = stype_bias.to_dict(orient="records")

    # 5. Residual distribution plot
    plot_residual_distribution(
        df, metric_col, human_col,
        results_dir / "figures" / f"residual_{metric_name}_{human_col}.png",
        is_mock=is_mock,
    )

    return results
