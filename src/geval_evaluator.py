"""
src/geval_evaluator.py
======================
G-Eval: NLG evaluation using LLMs with probability-weighted scoring.

Implements the EMNLP 2023 paper "G-Eval: NLG Evaluation using GPT-4 with
Better Human Alignment" (Liu et al.) faithfully:

1. Four dimension prompts (coherence, consistency, fluency, relevance)
2. Auto-CoT: generate evaluation steps once, cache in prompts/auto_cot_*.txt
3. Probability-weighted score from token logprobs (or n=20 sampling fallback)
4. Ablation variants: full | no-probs | no-cot | no-description

Deviations from the paper are logged to docs/DECISIONS.md.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm.asyncio import tqdm as atqdm

from src.llm_clients import BaseLLMClient, MockLLMClient
from src.utils import (
    DIMENSIONS,
    PROJECT_ROOT,
    SCORE_RANGES,
    CostTracker,
    parse_score,
)

logger = logging.getLogger(__name__)

# Prompt directory
PROMPTS_DIR = PROJECT_ROOT / "prompts"
AUTO_COT_DIR = PROMPTS_DIR


# ---------------------------------------------------------------------------
# Ablation mode registry
# ---------------------------------------------------------------------------

ABLATION_MODES = {
    "full": "Full G-Eval (CoT + prob-weighted)",
    "no_probs": "No probability weighting (temp=0, argmax)",
    "no_cot": "No chain-of-thought (bare scoring steps removed)",
    "no_description": "No dimension description (bare prompt)",
}


# ---------------------------------------------------------------------------
# Prompt management
# ---------------------------------------------------------------------------

def load_prompt_template(dimension: str, ablation: str = "full") -> str:
    """Load a prompt template for a given dimension and ablation variant.

    Args:
        dimension: One of coherence, consistency, fluency, relevance.
        ablation:  One of full, no_cot, no_description.

    Returns:
        Prompt string with {{Document}}, {{Summary}}, {{AutoCoT}} placeholders.
    """
    if ablation == "no_cot":
        path = PROMPTS_DIR / "ablation" / f"no_cot_{dimension}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        # Fall back to creating on-the-fly
        full_template = (PROMPTS_DIR / f"{dimension}.txt").read_text(encoding="utf-8")
        # Remove the AutoCoT section
        return re.sub(r"\{\{AutoCoT\}\}\n?", "", full_template)
    elif ablation == "no_description":
        path = PROMPTS_DIR / "ablation" / f"no_desc_{dimension}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        # Minimal prompt
        min_s, max_s = SCORE_RANGES[dimension]
        return (
            f"Source Text:\n\n{{{{Document}}}}\n\nSummary:\n\n{{{{Summary}}}}\n\n"
            f"Rate the {dimension} of the summary on a scale of {min_s} to {max_s}.\n\n"
            f"Evaluation Form (scores ONLY):\n\n- {dimension.capitalize()}:"
        )
    else:  # full
        return (PROMPTS_DIR / f"{dimension}.txt").read_text(encoding="utf-8")


def get_auto_cot_path(dimension: str) -> Path:
    """Return the cache path for a dimension's auto-CoT steps."""
    return AUTO_COT_DIR / f"auto_cot_{dimension}.txt"


def load_or_generate_cot(dimension: str) -> str:
    """Load cached CoT steps, or return a placeholder for generation."""
    path = get_auto_cot_path(dimension)
    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        if content and not content.startswith("[Generated at runtime:"):
            return content
    return ""  # Will be generated lazily


async def generate_auto_cot(
    dimension: str,
    client: BaseLLMClient,
    force_regenerate: bool = False,
) -> str:
    """Generate and cache chain-of-thought evaluation steps for a dimension.

    If the cached file already exists, returns its content.
    If the client is a mock, returns canonical placeholder steps.
    """
    path = get_auto_cot_path(dimension)
    if path.exists() and not force_regenerate:
        cached_steps = path.read_text(encoding="utf-8").strip()
        if cached_steps and not cached_steps.startswith("[Generated at runtime:"):
            logger.info(f"Auto-CoT for {dimension} loaded from {path}")
            return cached_steps

    if isinstance(client, MockLLMClient):
        return _get_mock_cot_steps(dimension)

    cot_prompt = _make_cot_generation_prompt(dimension)
    logger.info(f"Generating auto-CoT for dimension: {dimension}")
    resp = await client.complete(cot_prompt, temperature=0.0, max_tokens=512)
    cot_steps = resp.text.strip()
    path.write_text(cot_steps, encoding="utf-8")
    logger.info(f"Saved auto-CoT for {dimension} to {path}")
    return cot_steps


def _make_cot_generation_prompt(dimension: str) -> str:
    """Create the prompt to ask the LLM to generate evaluation steps."""
    min_s, max_s = SCORE_RANGES[dimension]
    template = (
        PROMPTS_DIR / "cot_generation" / f"{dimension}.txt"
    ).read_text(encoding="utf-8")
    return (
        template.replace("{{Dimension}}", dimension)
        .replace("{{MinScore}}", str(min_s))
        .replace("{{MaxScore}}", str(max_s))
    )


def _get_mock_cot_steps(dimension: str) -> str:
    """Return canonical mock evaluation steps for each dimension."""
    steps = {
        "coherence": (
            "1. Read the news article carefully and identify the main topic and key points.\n"
            "2. Read the summary and check if it covers the main topic in a logical order.\n"
            "3. Assess whether the sentences flow naturally from one to the next.\n"
            "4. Check if the summary avoids redundancy and irrelevant information.\n"
            "5. Assign a coherence score from 1 (incoherent) to 5 (highly coherent)."
        ),
        "consistency": (
            "1. Read the news article carefully and identify the main facts.\n"
            "2. Read the summary and check each claim against the source document.\n"
            "3. Identify any facts that are not supported by or contradict the source.\n"
            "4. Penalize summaries that contain hallucinated or fabricated information.\n"
            "5. Assign a consistency score from 1 (highly inconsistent) to 5 (fully consistent)."
        ),
        "fluency": (
            "1. Read the summary carefully.\n"
            "2. Check for grammatical errors, spelling mistakes, and punctuation issues.\n"
            "3. Assess whether word choices are appropriate and natural.\n"
            "4. Evaluate whether sentences are well-formed and easy to read.\n"
            "5. Assign a fluency score: 1 (poor), 2 (fair), or 3 (good)."
        ),
        "relevance": (
            "1. Read the news article and identify the most important information.\n"
            "2. Read the summary and check if it focuses on the key points.\n"
            "3. Identify any redundant, repetitive, or extraneous information.\n"
            "4. Assess whether the summary captures the most salient aspects.\n"
            "5. Assign a relevance score from 1 (not relevant) to 5 (highly relevant)."
        ),
    }
    return steps.get(dimension, f"1. Evaluate the {dimension} of the summary.\n2. Assign a score.")


def fill_prompt(
    template: str,
    document: str,
    summary: str,
    cot_steps: str = "",
) -> str:
    """Fill a prompt template with the actual text inputs."""
    prompt = template
    prompt = prompt.replace("{{Document}}", document)
    prompt = prompt.replace("{{Summary}}", summary)
    prompt = prompt.replace("{{AutoCoT}}", cot_steps)
    return prompt


# ---------------------------------------------------------------------------
# Core G-Eval Evaluator
# ---------------------------------------------------------------------------

class GEvalEvaluator:
    """G-Eval evaluator for NLG text quality assessment.

    Supports four dimensions (coherence, consistency, fluency, relevance)
    and three ablation variants.
    """

    def __init__(
        self,
        client: BaseLLMClient,
        dimensions: Optional[List[str]] = None,
        score_mode: str = "logprobs",  # logprobs | sampling
        n_samples: int = 20,
        ablation: str = "full",
        results_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> None:
        self.client = client
        self.dimensions = dimensions or list(DIMENSIONS)
        self.score_mode = score_mode
        self.n_samples = n_samples
        self.ablation = ablation
        self.is_mock = isinstance(client, MockLLMClient)
        self.results_dir = results_dir or (
            PROJECT_ROOT / "results" / "_mock" if self.is_mock
            else PROJECT_ROOT / "results"
        )
        self.dry_run = dry_run
        self._cot_cache: Dict[str, str] = {}

    async def setup_cot(self) -> None:
        """Pre-generate or load auto-CoT steps for all dimensions."""
        for dim in self.dimensions:
            self._cot_cache[dim] = await generate_auto_cot(dim, self.client)
        logger.info("Auto-CoT ready for all dimensions.")

    async def evaluate_record(
        self,
        source: str,
        summary: str,
        dimension: str,
        use_cot: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a single (source, summary) pair on one dimension.

        Returns:
            dict with keys: score, weighted_score, logprob_weights, raw_text, cached, is_mock.
        """
        min_s, max_s = SCORE_RANGES[dimension]
        allowed_scores = list(range(int(min_s), int(max_s) + 1))

        template = load_prompt_template(dimension, ablation=self.ablation)
        cot_steps = self._cot_cache.get(dimension, "") if use_cot else ""
        prompt = fill_prompt(template, source, summary, cot_steps)

        if self.dry_run:
            logger.info(f"[DRY RUN] Would evaluate {dimension} for summary[:50]={summary[:50]!r}")
            return {
                "score": float("nan"),
                "weighted_score": float("nan"),
                "logprob_weights": {},
                "raw_text": "[DRY RUN]",
                "cached": False,
                "is_mock": self.is_mock,
            }

        if self.ablation == "no_probs":
            resp = await self.client.complete(
                prompt,
                temperature=0.0,
                max_tokens=5,
                min_score=min_s,
                max_score=max_s,
            )
            resp.weighted_score = resp.score
        elif self.score_mode == "logprobs":
            resp = await self.client.complete_with_logprobs(
                prompt, allowed_scores=allowed_scores,
                temperature=1.0, max_tokens=5, top_p=1.0,
                min_score=min_s, max_score=max_s,
            )
        else:
            resp = await self.client.sample_n(
                prompt, n=self.n_samples,
                temperature=1.0, max_tokens=5, top_p=1.0,
                min_score=min_s, max_score=max_s,
            )

        if resp.score is None:
            resp.score = parse_score(resp.text, min_score=min_s, max_score=max_s)
        if resp.weighted_score is None:
            resp.weighted_score = resp.score
        if resp.weighted_score is None or not math.isfinite(resp.weighted_score):
            raise ValueError(f"Could not obtain a valid {dimension} score from {resp.text!r}")
        if not min_s <= resp.weighted_score <= max_s:
            raise ValueError(
                f"{dimension} score {resp.weighted_score} is outside [{min_s}, {max_s}]"
            )

        return {
            "score": resp.score,
            "weighted_score": resp.weighted_score,
            "logprob_weights": resp.logprob_weights,
            "raw_text": resp.text,
            "cached": resp.cached,
            "is_mock": resp.is_mock,
        }

    async def evaluate_dataset(
        self,
        records: List[Dict[str, Any]],
        limit: Optional[int] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Evaluate all records across all dimensions.

        Args:
            records:       List of SummEval records.
            limit:         Subsample (for smoke tests).
            show_progress: Show tqdm progress bar.

        Returns:
            DataFrame with columns: doc_id, system_id, dimension, score,
            weighted_score, is_mock, cached.
        """
        if limit is not None:
            records = records[:limit]

        await self.setup_cot()

        # Build tasks
        tasks = []
        meta = []
        for rec in records:
            for dim in self.dimensions:
                tasks.append(
                    self.evaluate_record(
                        source=rec["source"],
                        summary=rec["system_output"],
                        dimension=dim,
                        use_cot=(self.ablation not in ("no_cot", "no_description")),
                    )
                )
                meta.append({
                    "doc_id": rec["doc_id"],
                    "system_id": rec["system_id"],
                    "dimension": dim,
                })

        # Run with progress bar
        if show_progress:
            results_raw = await atqdm.gather(*tasks, desc=f"G-Eval [{self.ablation}]")
        else:
            results_raw = await asyncio.gather(*tasks)

        rows = []
        for m, r in zip(meta, results_raw):
            rows.append({
                **m,
                "score": r["score"],
                "weighted_score": r["weighted_score"],
                "raw_text": r["raw_text"],
                "cached": r["cached"],
                "is_mock": r["is_mock"],
            })

        df = pd.DataFrame(rows)

        # Add overall = mean of four dimensions (per record)
        df_pivot = df.pivot_table(
            index=["doc_id", "system_id"],
            columns="dimension",
            values="weighted_score",
            aggfunc="first",
        ).reset_index()
        dim_cols = [d for d in self.dimensions if d in df_pivot.columns]
        df_pivot["overall"] = df_pivot[dim_cols].mean(axis=1)

        # Merge back
        overall_rows = []
        for _, row in df_pivot.iterrows():
            overall_rows.append({
                "doc_id": row["doc_id"],
                "system_id": row["system_id"],
                "dimension": "overall",
                "score": row["overall"],
                "weighted_score": row["overall"],
                "raw_text": "mean",
                "cached": False,
                "is_mock": self.is_mock,
            })
        df = pd.concat([df, pd.DataFrame(overall_rows)], ignore_index=True)

        return df

    def estimate_cost(
        self,
        n_records: int,
        avg_prompt_tokens: int = 500,
        avg_output_tokens: int = 5,
    ) -> str:
        """Estimate API cost for evaluating n_records."""
        if self.client.provider == "mock":
            return f"Mock mode: {n_records:,} records, 0 API calls, $0.00 API cost."

        n_calls_per_record = len(self.dimensions)
        if self.score_mode == "sampling" or self.client.provider not in {
            "openai",
        }:
            n_calls_per_record *= self.n_samples
        total_calls = n_records * n_calls_per_record
        total_input = total_calls * avg_prompt_tokens
        total_output = total_calls * avg_output_tokens
        tracker = CostTracker(self.client.model)
        tracker.log(total_input, total_output)
        return (
            f"Estimated cost for {n_records} records × {n_calls_per_record} calls "
            f"= {total_calls:,} API calls. {tracker.report()}"
        )
