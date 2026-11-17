"""
src/traditional_metrics.py
==========================
Compute ROUGE, BLEU, and BERTScore for SummEval outputs.

Metrics:
  - ROUGE-1/2/L (F1) vs reference summary (rouge-score library)
  - ROUGE-1/2/L (F1) vs source article (reference-free)
  - BLEU (SacreBLEU, sentence-level) vs reference
  - BERTScore (P/R/F1) vs reference (roberta-large → distilbert fallback)
  - BERTScore (P/R/F1) vs source (reference-free)

Results are cached to results/traditional_metrics.csv.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.utils import (
    RESULTS_DIR,
    ensure_results_dir,
    records_to_dataframe,
    save_results,
    load_results,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

def compute_rouge(
    hypothesis: str,
    reference: str,
) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except ImportError:
        raise RuntimeError("rouge-score is required; install the pinned requirements.") from None


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def compute_bleu(hypothesis: str, reference: str) -> float:
    """Compute sentence-level BLEU score using SacreBLEU."""
    try:
        import sacrebleu
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
        return float(bleu.score) / 100.0  # normalize to [0, 1]
    except ImportError:
        raise RuntimeError("sacrebleu is required; install the pinned requirements.") from None


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def _get_bertscore_model(
    preferred: str = "roberta-large",
    fallback: str = "distilbert-base-uncased",
) -> str:
    """Determine the best BERTScore model available."""
    import torch
    if torch.cuda.is_available():
        model = preferred
    else:
        logger.warning(
            "No GPU detected. Using %s instead of %s for CPU-based BERTScore.",
            fallback,
            preferred,
        )
        model = fallback
    logger.info("BERTScore model: %s", model)
    return model


def compute_bertscore_batch(
    hypotheses: List[str],
    references: List[str],
    model_type: Optional[str] = None,
    batch_size: int = 32,
    lang: str = "en",
) -> Tuple[List[float], List[float], List[float]]:
    """Compute BERTScore for a batch of (hypothesis, reference) pairs.

    Returns:
        Tuple of (precision, recall, f1) lists.
    """
    if len(hypotheses) != len(references):
        raise ValueError("Hypotheses and references must have the same length")
    if not hypotheses:
        return [], [], []

    try:
        from bert_score import score as bert_score_fn
    except ImportError as exc:
        raise RuntimeError("bert-score is required; install the pinned requirements.") from exc

    selected_model = model_type or _get_bertscore_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision, recall, f1 = bert_score_fn(
            hypotheses,
            references,
            model_type=selected_model,
            lang=lang,
            batch_size=batch_size,
            verbose=False,
        )
    return precision.tolist(), recall.tolist(), f1.tolist()


# ---------------------------------------------------------------------------
# Main computation pipeline
# ---------------------------------------------------------------------------

def compute_all_traditional_metrics(
    records: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    bertscore_model: Optional[str] = None,
    bertscore_preferred_model: str = "roberta-large",
    bertscore_fallback_model: str = "distilbert-base-uncased",
    bertscore_batch_size: int = 32,
    limit: Optional[int] = None,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute all traditional metrics for all SummEval records.

    Args:
        records:            SummEval records list.
        output_path:        Where to save the CSV. If None, uses default.
        bertscore_model:    Override BERTScore model.
        bertscore_batch_size: Batch size for BERTScore.
        limit:              Subsample for testing.
        results_dir:        Override results directory.

    Returns:
        DataFrame with all metric scores.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    records = records[:limit] if limit is not None else records

    res_dir = results_dir or RESULTS_DIR
    ensure_results_dir(res_dir)
    cache_path = output_path or res_dir / "traditional_metrics.csv"
    model = bertscore_model or _get_bertscore_model(
        bertscore_preferred_model,
        bertscore_fallback_model,
    )

    requested_pairs = [(record["doc_id"], record["system_id"]) for record in records]
    if cache_path.exists():
        cached_df = load_results(cache_path)
        required_columns = {"doc_id", "system_id", "bertscore_model", *TRADITIONAL_METRIC_COLS}
        if required_columns.issubset(cached_df.columns) and cached_df["bertscore_model"].eq(model).all():
            cached_pairs = set(zip(cached_df["doc_id"], cached_df["system_id"]))
            if set(requested_pairs).issubset(cached_pairs):
                logger.info("Loading cached traditional metrics from %s", cache_path)
                indexed = cached_df.set_index(["doc_id", "system_id"])
                return indexed.loc[requested_pairs].reset_index()

    df = records_to_dataframe(records)
    n = len(df)
    logger.info(f"Computing traditional metrics for {n} records...")

    # ---- ROUGE ----
    logger.info("Computing ROUGE scores...")
    rouge1_ref, rouge2_ref, rougeL_ref = [], [], []
    rouge1_src, rouge2_src, rougeL_src = [], [], []
    bleu_ref = []

    for _, row in tqdm(df.iterrows(), total=n, desc="ROUGE+BLEU"):
        hyp = row["system_output"]
        ref = row["reference"]
        src = row["source"]

        r_ref = compute_rouge(hyp, ref)
        rouge1_ref.append(r_ref["rouge1"])
        rouge2_ref.append(r_ref["rouge2"])
        rougeL_ref.append(r_ref["rougeL"])

        r_src = compute_rouge(hyp, src)
        rouge1_src.append(r_src["rouge1"])
        rouge2_src.append(r_src["rouge2"])
        rougeL_src.append(r_src["rougeL"])

        bleu_ref.append(compute_bleu(hyp, ref))

    df["rouge1_ref"] = rouge1_ref
    df["rouge2_ref"] = rouge2_ref
    df["rougeL_ref"] = rougeL_ref
    df["rouge1_src"] = rouge1_src
    df["rouge2_src"] = rouge2_src
    df["rougeL_src"] = rougeL_src
    df["bleu_ref"] = bleu_ref

    # ---- BERTScore vs reference ----
    logger.info("Computing BERTScore vs reference...")
    hyps = df["system_output"].tolist()
    refs = df["reference"].tolist()
    srcs = df["source"].tolist()

    P_ref, R_ref, F_ref = compute_bertscore_batch(hyps, refs, model_type=model,
                                                   batch_size=bertscore_batch_size)
    df["bertscore_P_ref"] = P_ref
    df["bertscore_R_ref"] = R_ref
    df["bertscore_F1_ref"] = F_ref

    # ---- BERTScore vs source (reference-free) ----
    logger.info("Computing BERTScore vs source (reference-free)...")
    P_src, R_src, F_src = compute_bertscore_batch(hyps, srcs, model_type=model,
                                                   batch_size=bertscore_batch_size)
    df["bertscore_P_src"] = P_src
    df["bertscore_R_src"] = R_src
    df["bertscore_F1_src"] = F_src

    # Drop heavy text columns before saving
    cols_to_keep = [
        "doc_id", "system_id",
        "human_coherence", "human_consistency", "human_fluency",
        "human_relevance", "human_overall",
        "rouge1_ref", "rouge2_ref", "rougeL_ref",
        "rouge1_src", "rouge2_src", "rougeL_src",
        "bleu_ref",
        "bertscore_P_ref", "bertscore_R_ref", "bertscore_F1_ref",
        "bertscore_P_src", "bertscore_R_src", "bertscore_F1_src",
    ]
    result_df = df[cols_to_keep].copy()
    result_df["bertscore_model"] = model
    save_results(result_df, cache_path)
    logger.info(f"Traditional metrics saved to {cache_path}")
    return result_df


TRADITIONAL_METRIC_COLS = [
    "rouge1_ref", "rouge2_ref", "rougeL_ref",
    "rouge1_src", "rouge2_src", "rougeL_src",
    "bleu_ref",
    "bertscore_P_ref", "bertscore_R_ref", "bertscore_F1_ref",
    "bertscore_P_src", "bertscore_R_src", "bertscore_F1_src",
]
