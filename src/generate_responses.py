"""
src/generate_responses.py
=========================
Optional: Generate fresh summaries from multiple LLMs for a 100-article subset.
Only runs if config.generate_responses.enabled = true and API keys are available.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm.asyncio import tqdm as atqdm

from src.llm_clients import BaseLLMClient, create_client, DiskCache
from src.utils import PROJECT_ROOT, ensure_results_dir, load_config

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_PROMPT_TEMPLATE = (
    "Please write a concise, accurate summary of the following news article in 3-4 sentences:\n\n{article}"
)


async def generate_summary(
    client: BaseLLMClient,
    article: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    """Generate a summary for one article."""
    prompt = prompt_template.format(article=article)
    resp = await client.complete(prompt, temperature=0.7, max_tokens=200)
    return resp.text.strip()


async def generate_all_summaries(
    articles: List[Dict[str, Any]],
    models: List[Dict[str, str]],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    cache_db: Optional[str] = None,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate summaries for all articles × all models.

    Args:
        articles:        List of dicts with doc_id and source.
        models:          List of {provider, model, label} dicts.
        prompt_template: Format string with {article}.
        cache_db:        Path to SQLite cache.
        results_dir:     Output directory.

    Returns:
        DataFrame with doc_id, provider, model, label, summary.
    """
    res_dir = results_dir or RESULTS_DIR
    ensure_results_dir(res_dir)
    output_path = res_dir / "generated_summaries.csv"
    if output_path.exists():
        logger.info(f"Generated summaries already exist at {output_path}. Loading.")
        return pd.read_csv(output_path)

    cache = DiskCache(cache_db) if cache_db else None
    rows = []

    for model_cfg in models:
        provider = model_cfg["provider"]
        model = model_cfg["model"]
        label = model_cfg.get("label", f"{provider}/{model}")
        logger.info(f"Generating summaries with {label}...")

        client = create_client(provider=provider, model=model, cache=cache)

        async def gen(article_dict):
            summary = await generate_summary(client, article_dict["source"], prompt_template)
            return {
                "doc_id": article_dict["doc_id"],
                "provider": provider,
                "model": model,
                "label": label,
                "summary": summary,
            }

        tasks = [gen(a) for a in articles]
        results = await atqdm.gather(*tasks, desc=f"Generating [{label}]")
        rows.extend(results)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Generated summaries saved to {output_path}")
    return df


def run_generation(config: Optional[Dict] = None) -> Optional[pd.DataFrame]:
    """Entry point: check config and run generation if enabled."""
    if config is None:
        config = load_config()

    gen_cfg = config.get("generate_responses", {})
    if not gen_cfg.get("enabled", False):
        logger.info("Fresh response generation is disabled (set generate_responses.enabled=true in config).")
        return None

    models = gen_cfg.get("models", [])
    if not models:
        logger.warning("No models configured for response generation.")
        return None

    n_articles = gen_cfg.get("n_articles", 100)
    prompt_template = gen_cfg.get("prompt", DEFAULT_PROMPT_TEMPLATE)

    # Load articles
    from src.utils import load_summeval
    data = load_summeval()
    # Get unique articles
    seen = set()
    articles = []
    for rec in data:
        if rec["doc_id"] not in seen:
            seen.add(rec["doc_id"])
            articles.append({"doc_id": rec["doc_id"], "source": rec["source"]})
        if len(articles) >= n_articles:
            break

    logger.info(f"Generating summaries for {len(articles)} articles × {len(models)} models...")
    df = asyncio.run(generate_all_summaries(
        articles=articles,
        models=models,
        prompt_template=prompt_template,
        cache_db=config.get("cache_db"),
    ))
    return df
