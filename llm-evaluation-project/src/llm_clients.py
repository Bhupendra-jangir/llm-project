"""
src/llm_clients.py
==================
Provider-agnostic LLM client adapters.

Supported providers:
  - mock       : MockLLMClient (deterministic, no key needed)
  - openai     : OpenAI GPT models
  - gemini     : Google Gemini
  - anthropic  : Anthropic Claude
  - groq       : Groq (OpenAI-compatible)
  - openrouter : OpenRouter (OpenAI-compatible)
  - ollama     : Local Ollama

All clients expose the same async interface:
    client.complete(prompt, **kwargs) -> LLMResponse
    client.complete_with_logprobs(prompt, scores, **kwargs) -> LLMResponse

Engineering features (applied by all real clients via BaseLLMClient):
  - Async with asyncio.Semaphore concurrency limit
  - Exponential-backoff retries via tenacity
  - Disk cache (SQLite via DiskCache)
  - Token / cost logging via CostTracker
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils import CostTracker, DiskCache, parse_score

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Structured response from any LLM client."""
    text: str                           # Raw text response
    score: Optional[float] = None       # Parsed numeric score
    weighted_score: Optional[float] = None  # Probability-weighted score
    logprob_weights: Dict[str, float] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""
    cached: bool = False
    is_mock: bool = False


# ---------------------------------------------------------------------------
# Base client
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """Abstract base with caching, retries, and semaphore."""

    def __init__(
        self,
        provider: str,
        model: str,
        cache: Optional[DiskCache] = None,
        cost_tracker: Optional[CostTracker] = None,
        concurrency_limit: int = 5,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.cache = cache
        self.cost_tracker = cost_tracker or CostTracker(model)
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def _cache_key(self, prompt: str, **kwargs: Any) -> str:
        return DiskCache.make_key(
            provider=self.provider,
            model=self.model,
            prompt=prompt,
            **{k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))},
        )

    async def complete(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 5,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion (single response)."""
        key = self._cache_key(prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        if self.cache and key in self.cache:
            cached = self.cache.get(key)
            resp = LLMResponse(**cached, cached=True)
            return resp

        async with self._semaphore:
            resp = await self._complete_with_retry(
                prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs
            )

        if self.cache and not resp.is_mock:
            self.cache.set(key, {
                "text": resp.text,
                "score": resp.score,
                "weighted_score": resp.weighted_score,
                "logprob_weights": resp.logprob_weights,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "model": resp.model,
                "provider": resp.provider,
                "is_mock": resp.is_mock,
            })
        if self.cost_tracker:
            self.cost_tracker.log(resp.input_tokens, resp.output_tokens)
        return resp

    async def complete_with_logprobs(
        self,
        prompt: str,
        allowed_scores: Sequence[int],
        temperature: float = 1.0,
        max_tokens: int = 5,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion with token logprob-weighted scoring."""
        key = self._cache_key(
            prompt,
            mode="logprobs",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        if self.cache and key in self.cache:
            cached = self.cache.get(key)
            return LLMResponse(**cached, cached=True)

        async with self._semaphore:
            resp = await self._complete_logprobs_with_retry(
                prompt,
                allowed_scores=allowed_scores,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )

        if self.cache and not resp.is_mock:
            self.cache.set(key, {
                "text": resp.text,
                "score": resp.score,
                "weighted_score": resp.weighted_score,
                "logprob_weights": resp.logprob_weights,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "model": resp.model,
                "provider": resp.provider,
                "is_mock": resp.is_mock,
            })
        if self.cost_tracker:
            self.cost_tracker.log(resp.input_tokens, resp.output_tokens)
        return resp

    async def sample_n(
        self,
        prompt: str,
        n: int = 20,
        temperature: float = 1.0,
        max_tokens: int = 5,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Sample n completions and return frequency-weighted score."""
        key = self._cache_key(
            prompt, mode="sampling", n=n, temperature=temperature, max_tokens=max_tokens
        )
        if self.cache and key in self.cache:
            cached = self.cache.get(key)
            return LLMResponse(**cached, cached=True)

        async with self._semaphore:
            resp = await self._sample_n_with_retry(
                prompt, n=n, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs
            )

        if self.cache and not resp.is_mock:
            self.cache.set(key, {
                "text": resp.text,
                "score": resp.score,
                "weighted_score": resp.weighted_score,
                "logprob_weights": resp.logprob_weights,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "model": resp.model,
                "provider": resp.provider,
                "is_mock": resp.is_mock,
            })
        if self.cost_tracker:
            self.cost_tracker.log(resp.input_tokens, resp.output_tokens)
        return resp

    async def _complete_with_retry(self, prompt: str, **kwargs: Any) -> LLMResponse:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.retry_base_delay, min=1, max=60),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                return await self._complete(prompt, **kwargs)

    async def _complete_logprobs_with_retry(self, prompt: str, **kwargs: Any) -> LLMResponse:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.retry_base_delay, min=1, max=60),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                return await self._complete_with_logprobs(prompt, **kwargs)

    async def _sample_n_with_retry(self, prompt: str, **kwargs: Any) -> LLMResponse:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.retry_base_delay, min=1, max=60),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                return await self._sample_n(prompt, **kwargs)

    @abstractmethod
    async def _complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        ...

    @abstractmethod
    async def _complete_with_logprobs(
        self, prompt: str, allowed_scores: Sequence[int], **kwargs: Any
    ) -> LLMResponse:
        ...

    @abstractmethod
    async def _sample_n(self, prompt: str, n: int, **kwargs: Any) -> LLMResponse:
        ...

    @staticmethod
    def _weighted_score_from_logprobs(
        logprobs: Dict[str, float], allowed_scores: Sequence[int]
    ) -> Tuple[float, Dict[str, float]]:
        """Compute probability-weighted score from token logprobs."""
        # Convert logprobs to probabilities
        weights: Dict[str, float] = {}
        for score_val in allowed_scores:
            token = str(score_val)
            lp = logprobs.get(token, -100.0)
            weights[token] = math.exp(lp)

        total = sum(weights.values())
        if total == 0:
            return float(np.mean(allowed_scores)), {}
        norm = {k: v / total for k, v in weights.items()}
        weighted = sum(float(k) * v for k, v in norm.items())
        return weighted, norm

    @staticmethod
    def _weighted_score_from_samples(
        texts: List[str], allowed_scores: Sequence[int], min_score: float, max_score: float
    ) -> Tuple[float, Dict[str, float]]:
        """Compute frequency-weighted score from sampled texts."""
        counts: Dict[str, int] = {str(s): 0 for s in allowed_scores}
        total = 0
        for text in texts:
            score = parse_score(text, min_score=min_score, max_score=max_score)
            if score is not None:
                key = str(int(round(score)))
                if key in counts:
                    counts[key] += 1
                    total += 1

        if total == 0:
            return float(np.mean(allowed_scores)), {}

        weights = {k: v / total for k, v in counts.items()}
        weighted = sum(float(k) * v for k, v in weights.items())
        return weighted, weights


# ---------------------------------------------------------------------------
# MockLLMClient — deterministic, no API key
# ---------------------------------------------------------------------------

class MockLLMClient(BaseLLMClient):
    """Deterministic mock LLM client for testing and smoke runs.

    Scores are seeded random values based on the hash of the prompt,
    so they are deterministic across runs but vary by prompt.

    IMPORTANT: Mock outputs must NEVER be written to real results/.
    They should go to results/_mock/.
    """

    IS_MOCK = True

    def __init__(self, model: str = "mock-model", seed: int = 42, **kwargs: Any) -> None:
        super().__init__(provider="mock", model=model, **kwargs)
        self.seed = seed

    def _prompt_seed(self, prompt: str) -> int:
        """Generate a deterministic integer seed from a prompt."""
        h = hashlib.md5(prompt.encode()).hexdigest()
        return int(h[:8], 16) ^ self.seed

    def _mock_score(self, prompt: str, min_s: float = 1.0, max_s: float = 5.0) -> float:
        rng = np.random.default_rng(self._prompt_seed(prompt))
        return float(rng.uniform(min_s, max_s))

    def _mock_logprobs(
        self, prompt: str, allowed_scores: Sequence[int]
    ) -> Dict[str, float]:
        rng = np.random.default_rng(self._prompt_seed(prompt))
        raw = rng.dirichlet(np.ones(len(allowed_scores)))
        return {str(s): float(np.log(p + 1e-10)) for s, p in zip(allowed_scores, raw)}

    async def _complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        await asyncio.sleep(0.001)  # Simulate minimal latency
        score = self._mock_score(prompt)
        return LLMResponse(
            text=f"{score:.1f}",
            score=score,
            weighted_score=score,
            model=self.model,
            provider="mock",
            input_tokens=len(prompt.split()),
            output_tokens=1,
            is_mock=True,
        )

    async def _complete_with_logprobs(
        self, prompt: str, allowed_scores: Sequence[int], **kwargs: Any
    ) -> LLMResponse:
        await asyncio.sleep(0.001)
        logprobs = self._mock_logprobs(prompt, allowed_scores)
        min_s = min(allowed_scores)
        max_s = max(allowed_scores)
        weighted, weights = self._weighted_score_from_logprobs(logprobs, allowed_scores)
        return LLMResponse(
            text=str(int(round(weighted))),
            score=round(weighted),
            weighted_score=weighted,
            logprob_weights=weights,
            model=self.model,
            provider="mock",
            input_tokens=len(prompt.split()),
            output_tokens=1,
            is_mock=True,
        )

    async def _sample_n(self, prompt: str, n: int = 20, **kwargs: Any) -> LLMResponse:
        await asyncio.sleep(0.001)
        rng = np.random.default_rng(self._prompt_seed(prompt))
        # Guess the scale from kwargs or default to 1-5
        min_s = float(kwargs.get("min_score", 1))
        max_s = float(kwargs.get("max_score", 5))
        allowed_scores = list(range(int(min_s), int(max_s) + 1))
        texts = [str(int(rng.integers(int(min_s), int(max_s) + 1))) for _ in range(n)]
        weighted, weights = self._weighted_score_from_samples(
            texts, allowed_scores, min_s, max_s
        )
        return LLMResponse(
            text="|".join(texts[:3]) + "...",
            score=round(weighted),
            weighted_score=weighted,
            logprob_weights=weights,
            model=self.model,
            provider="mock",
            input_tokens=len(prompt.split()),
            output_tokens=n,
            is_mock=True,
        )


# ---------------------------------------------------------------------------
# OpenAI Client
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client with logprob and sampling support."""

    def __init__(self, model: str = "gpt-4o", **kwargs: Any) -> None:
        super().__init__(provider="openai", model=model, **kwargs)
        try:
            import openai
            self._client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Install openai: pip install openai>=1.0")

    async def _complete(self, prompt: str, temperature: float = 1.0,
                        max_tokens: int = 5, top_p: float = 1.0, **kwargs: Any) -> LLMResponse:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        text = resp.choices[0].message.content or ""
        score = parse_score(text)
        usage = resp.usage
        return LLMResponse(
            text=text, score=score, weighted_score=score,
            model=self.model, provider="openai",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def _complete_with_logprobs(
        self, prompt: str, allowed_scores: Sequence[int],
        temperature: float = 1.0, max_tokens: int = 5, top_p: float = 1.0, **kwargs: Any
    ) -> LLMResponse:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            logprobs=True,
            top_logprobs=10,
        )
        text = resp.choices[0].message.content or ""
        # Extract logprobs for score tokens
        logprob_data: Dict[str, float] = {}
        if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
            for token_lp in resp.choices[0].logprobs.content[:3]:
                for top_lp in (token_lp.top_logprobs or []):
                    t = top_lp.token.strip()
                    if t in {str(s) for s in allowed_scores}:
                        if t not in logprob_data:
                            logprob_data[t] = top_lp.logprob

        if logprob_data:
            weighted, weights = self._weighted_score_from_logprobs(logprob_data, allowed_scores)
        else:
            # Fall back to parsing the text
            score = parse_score(text, min_score=min(allowed_scores), max_score=max(allowed_scores))
            weighted = score or float(np.mean(allowed_scores))
            weights = {}

        usage = resp.usage
        return LLMResponse(
            text=text, score=parse_score(text), weighted_score=weighted,
            logprob_weights=weights, model=self.model, provider="openai",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def _sample_n(
        self, prompt: str, n: int = 20, temperature: float = 1.0,
        max_tokens: int = 5, top_p: float = 1.0, **kwargs: Any
    ) -> LLMResponse:
        # OpenAI deprecated n>1 for chat; loop instead
        tasks = [
            self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            for _ in range(n)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        texts = []
        total_in, total_out = 0, 0
        for r in responses:
            if isinstance(r, Exception):
                continue
            texts.append(r.choices[0].message.content or "")
            if r.usage:
                total_in += r.usage.prompt_tokens
                total_out += r.usage.completion_tokens

        min_s = float(kwargs.get("min_score", 1))
        max_s = float(kwargs.get("max_score", 5))
        allowed_scores = list(range(int(min_s), int(max_s) + 1))
        weighted, weights = self._weighted_score_from_samples(texts, allowed_scores, min_s, max_s)
        return LLMResponse(
            text="|".join(texts[:3]), score=round(weighted), weighted_score=weighted,
            logprob_weights=weights, model=self.model, provider="openai",
            input_tokens=total_in, output_tokens=total_out,
        )


# ---------------------------------------------------------------------------
# Google Gemini Client
# ---------------------------------------------------------------------------

class GeminiClient(BaseLLMClient):
    """Google Gemini client."""

    def __init__(self, model: str = "gemini-1.5-flash", **kwargs: Any) -> None:
        super().__init__(provider="gemini", model=model, **kwargs)
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self._model = genai.GenerativeModel(model)
            self._genai = genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

    async def _complete(self, prompt: str, temperature: float = 1.0,
                        max_tokens: int = 5, top_p: float = 1.0, **kwargs: Any) -> LLMResponse:
        loop = asyncio.get_event_loop()
        gen_config = self._genai.types.GenerationConfig(
            temperature=temperature, max_output_tokens=max_tokens, top_p=top_p
        )
        resp = await loop.run_in_executor(
            None, lambda: self._model.generate_content(prompt, generation_config=gen_config)
        )
        text = resp.text if resp.text else ""
        score = parse_score(text)
        usage = resp.usage_metadata
        return LLMResponse(
            text=text, score=score, weighted_score=score,
            model=self.model, provider="gemini",
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
        )

    async def _complete_with_logprobs(
        self, prompt: str, allowed_scores: Sequence[int], **kwargs: Any
    ) -> LLMResponse:
        # Gemini doesn't support logprobs → fall back to sampling
        logger.warning("Gemini doesn't support logprobs; falling back to n=20 sampling.")
        return await self._sample_n(prompt, n=20, **kwargs)

    async def _sample_n(
        self, prompt: str, n: int = 20, temperature: float = 1.0,
        max_tokens: int = 5, top_p: float = 1.0, **kwargs: Any
    ) -> LLMResponse:
        loop = asyncio.get_event_loop()
        gen_config = self._genai.types.GenerationConfig(
            temperature=temperature, max_output_tokens=max_tokens, top_p=top_p,
            candidate_count=min(n, 8)  # Gemini caps candidates
        )
        resp = await loop.run_in_executor(
            None, lambda: self._model.generate_content(prompt, generation_config=gen_config)
        )
        texts = [c.content.parts[0].text for c in resp.candidates if c.content.parts]
        min_s = float(kwargs.get("min_score", 1))
        max_s = float(kwargs.get("max_score", 5))
        allowed = list(range(int(min_s), int(max_s) + 1))
        weighted, weights = self._weighted_score_from_samples(texts, allowed, min_s, max_s)
        usage = resp.usage_metadata
        return LLMResponse(
            text="|".join(texts[:3]), score=round(weighted), weighted_score=weighted,
            logprob_weights=weights, model=self.model, provider="gemini",
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
        )


# ---------------------------------------------------------------------------
# Anthropic Client
# ---------------------------------------------------------------------------

class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client."""

    def __init__(self, model: str = "claude-3-haiku-20240307", **kwargs: Any) -> None:
        super().__init__(provider="anthropic", model=model, **kwargs)
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    async def _complete(self, prompt: str, temperature: float = 1.0,
                        max_tokens: int = 5, top_p: float = 1.0, **kwargs: Any) -> LLMResponse:
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=max(max_tokens, 10),
            temperature=min(temperature, 1.0),
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text if resp.content else ""
        score = parse_score(text)
        return LLMResponse(
            text=text, score=score, weighted_score=score,
            model=self.model, provider="anthropic",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    async def _complete_with_logprobs(
        self, prompt: str, allowed_scores: Sequence[int], **kwargs: Any
    ) -> LLMResponse:
        logger.warning("Anthropic doesn't support logprobs; falling back to sampling.")
        return await self._sample_n(prompt, n=20, **kwargs)

    async def _sample_n(
        self, prompt: str, n: int = 20, temperature: float = 1.0,
        max_tokens: int = 5, **kwargs: Any
    ) -> LLMResponse:
        tasks = [self._complete(prompt, temperature=temperature, max_tokens=max_tokens)
                 for _ in range(n)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        texts = [r.text for r in responses if isinstance(r, LLMResponse)]
        min_s = float(kwargs.get("min_score", 1))
        max_s = float(kwargs.get("max_score", 5))
        allowed = list(range(int(min_s), int(max_s) + 1))
        weighted, weights = self._weighted_score_from_samples(texts, allowed, min_s, max_s)
        total_in = sum(r.input_tokens for r in responses if isinstance(r, LLMResponse))
        total_out = sum(r.output_tokens for r in responses if isinstance(r, LLMResponse))
        return LLMResponse(
            text="|".join(texts[:3]), score=round(weighted), weighted_score=weighted,
            logprob_weights=weights, model=self.model, provider="anthropic",
            input_tokens=total_in, output_tokens=total_out,
        )


# ---------------------------------------------------------------------------
# Groq / OpenRouter (OpenAI-compatible)
# ---------------------------------------------------------------------------

class OpenAICompatibleClient(BaseLLMClient):
    """Generic OpenAI-compatible client (Groq, OpenRouter)."""

    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key_env: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider=provider, model=model, **kwargs)
        try:
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=os.getenv(api_key_env, ""),
                base_url=base_url,
            )
        except ImportError:
            raise ImportError("Install openai: pip install openai>=1.0")

    async def _complete(self, prompt: str, temperature: float = 1.0,
                        max_tokens: int = 5, top_p: float = 1.0, **kwargs: Any) -> LLMResponse:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        text = resp.choices[0].message.content or ""
        score = parse_score(text)
        usage = resp.usage
        return LLMResponse(
            text=text, score=score, weighted_score=score,
            model=self.model, provider=self.provider,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def _complete_with_logprobs(
        self, prompt: str, allowed_scores: Sequence[int],
        temperature: float = 1.0, max_tokens: int = 5, top_p: float = 1.0, **kwargs: Any
    ) -> LLMResponse:
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                logprobs=True,
                top_logprobs=10,
            )
            text = resp.choices[0].message.content or ""
            logprob_data: Dict[str, float] = {}
            if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
                for token_lp in resp.choices[0].logprobs.content[:3]:
                    for top_lp in (token_lp.top_logprobs or []):
                        t = top_lp.token.strip()
                        if t in {str(s) for s in allowed_scores}:
                            if t not in logprob_data:
                                logprob_data[t] = top_lp.logprob
            if logprob_data:
                weighted, weights = self._weighted_score_from_logprobs(logprob_data, allowed_scores)
            else:
                score = parse_score(text, min_score=min(allowed_scores), max_score=max(allowed_scores))
                weighted = score or float(np.mean(allowed_scores))
                weights = {}
            usage = resp.usage
            return LLMResponse(
                text=text, score=parse_score(text), weighted_score=weighted,
                logprob_weights=weights, model=self.model, provider=self.provider,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )
        except Exception as e:
            logger.warning(f"Logprobs failed ({e}); falling back to sampling.")
            return await self._sample_n(prompt, n=20, allowed_scores=allowed_scores, **kwargs)

    async def _sample_n(
        self, prompt: str, n: int = 20, temperature: float = 1.0,
        max_tokens: int = 5, top_p: float = 1.0, **kwargs: Any
    ) -> LLMResponse:
        tasks = [
            self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            for _ in range(n)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        texts = []
        total_in, total_out = 0, 0
        for r in responses:
            if isinstance(r, Exception):
                continue
            texts.append(r.choices[0].message.content or "")
            if r.usage:
                total_in += r.usage.prompt_tokens
                total_out += r.usage.completion_tokens
        min_s = float(kwargs.get("min_score", 1))
        max_s = float(kwargs.get("max_score", 5))
        allowed = list(range(int(min_s), int(max_s) + 1))
        weighted, weights = self._weighted_score_from_samples(texts, allowed, min_s, max_s)
        return LLMResponse(
            text="|".join(texts[:3]), score=round(weighted), weighted_score=weighted,
            logprob_weights=weights, model=self.model, provider=self.provider,
            input_tokens=total_in, output_tokens=total_out,
        )


class GroqClient(OpenAICompatibleClient):
    def __init__(self, model: str = "llama-3.3-70b-versatile", **kwargs: Any) -> None:
        super().__init__(
            provider="groq",
            model=model,
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            **kwargs,
        )


class OpenRouterClient(OpenAICompatibleClient):
    def __init__(self, model: str = "openai/gpt-4o-mini", **kwargs: Any) -> None:
        super().__init__(
            provider="openrouter",
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Ollama Client
# ---------------------------------------------------------------------------

class OllamaClient(BaseLLMClient):
    """Local Ollama client."""

    def __init__(self, model: str = "llama3.2", **kwargs: Any) -> None:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        super().__init__(provider="ollama", model=model, **kwargs)
        self._base_url = host

    async def _post(self, endpoint: str, payload: Dict) -> Dict:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}{endpoint}", json=payload, timeout=aiohttp.ClientTimeout(total=120)
            ) as r:
                r.raise_for_status()
                return await r.json()

    async def _complete(self, prompt: str, temperature: float = 1.0,
                        max_tokens: int = 5, **kwargs: Any) -> LLMResponse:
        data = await self._post("/api/generate", {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        })
        text = data.get("response", "")
        score = parse_score(text)
        return LLMResponse(
            text=text, score=score, weighted_score=score,
            model=self.model, provider="ollama",
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    async def _complete_with_logprobs(
        self, prompt: str, allowed_scores: Sequence[int], **kwargs: Any
    ) -> LLMResponse:
        logger.warning("Ollama logprobs not supported; falling back to sampling.")
        return await self._sample_n(prompt, n=20, **kwargs)

    async def _sample_n(
        self, prompt: str, n: int = 20, temperature: float = 1.0,
        max_tokens: int = 5, **kwargs: Any
    ) -> LLMResponse:
        tasks = [self._complete(prompt, temperature=temperature, max_tokens=max_tokens)
                 for _ in range(n)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        texts = [r.text for r in responses if isinstance(r, LLMResponse)]
        min_s = float(kwargs.get("min_score", 1))
        max_s = float(kwargs.get("max_score", 5))
        allowed = list(range(int(min_s), int(max_s) + 1))
        weighted, weights = self._weighted_score_from_samples(texts, allowed, min_s, max_s)
        total_in = sum(r.input_tokens for r in responses if isinstance(r, LLMResponse))
        total_out = sum(r.output_tokens for r in responses if isinstance(r, LLMResponse))
        return LLMResponse(
            text="|".join(texts[:3]), score=round(weighted), weighted_score=weighted,
            logprob_weights=weights, model=self.model, provider="ollama",
            input_tokens=total_in, output_tokens=total_out,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_client(
    provider: str,
    model: str,
    cache: Optional[DiskCache] = None,
    cost_tracker: Optional[CostTracker] = None,
    concurrency_limit: int = 5,
    max_retries: int = 5,
    retry_base_delay: float = 1.0,
) -> BaseLLMClient:
    """Factory function to instantiate the appropriate LLM client."""
    common = dict(
        cache=cache,
        cost_tracker=cost_tracker,
        concurrency_limit=concurrency_limit,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )
    if provider == "mock":
        return MockLLMClient(model=model, **common)
    elif provider == "openai":
        return OpenAIClient(model=model, **common)
    elif provider == "gemini":
        return GeminiClient(model=model, **common)
    elif provider == "anthropic":
        return AnthropicClient(model=model, **common)
    elif provider == "groq":
        return GroqClient(model=model, **common)
    elif provider == "openrouter":
        return OpenRouterClient(model=model, **common)
    elif provider == "ollama":
        return OllamaClient(model=model, **common)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. "
                         f"Supported: mock, openai, gemini, anthropic, groq, openrouter, ollama")


def check_api_key_available(config: Dict[str, Any]) -> bool:
    """Check whether the configured provider has an API key in the environment."""
    provider = config.get("llm", {}).get("provider", "mock")
    env_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    if provider == "mock" or provider == "ollama":
        return True
    env_var = env_map.get(provider)
    if not env_var:
        return False
    val = os.getenv(env_var, "")
    return bool(val and not val.startswith("your_"))
