"""
tests/test_geval.py - Unit tests for G-Eval evaluator.
"""
import asyncio
import sys
import time
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_clients import MockLLMClient, BaseLLMClient, LLMResponse, OpenAIClient
from src.geval_evaluator import GEvalEvaluator, fill_prompt, generate_auto_cot


class TestMockClient:
    def test_deterministic(self):
        client = MockLLMClient(seed=42)
        async def run():
            r1 = await client.complete("test prompt")
            r2 = await client.complete("test prompt")
            return r1.weighted_score, r2.weighted_score
        s1, s2 = asyncio.run(run())
        assert s1 == s2

    def test_judge_models_have_distinct_seeded_outputs(self):
        first = MockLLMClient(model="judge-a", seed=42)
        second = MockLLMClient(model="judge-b", seed=42)
        assert first._mock_score("same prompt") != second._mock_score("same prompt")

    def test_logprobs_range(self):
        client = MockLLMClient(seed=42)
        async def run():
            return await client.complete_with_logprobs("test", allowed_scores=[1,2,3,4,5])
        resp = asyncio.run(run())
        assert resp.weighted_score is not None
        assert 1.0 <= resp.weighted_score <= 5.0
        assert resp.is_mock is True

    def test_sample_n_range(self):
        client = MockLLMClient(seed=42)
        async def run():
            return await client.sample_n("test", n=20, min_score=1, max_score=5)
        resp = asyncio.run(run())
        assert 1.0 <= resp.weighted_score <= 5.0


class TestPromptFilling:
    def test_all_placeholders_filled(self):
        template = "Document: {{Document}}\nSummary: {{Summary}}\nSteps: {{AutoCoT}}"
        result = fill_prompt(template, "doc text", "sum text", "1. Step one")
        assert "doc text" in result
        assert "sum text" in result
        assert "1. Step one" in result
        assert "{{" not in result

    def test_empty_cot_no_placeholder(self):
        template = "Document: {{Document}}\nSummary: {{Summary}}\nSteps: {{AutoCoT}}"
        result = fill_prompt(template, "doc", "sum", "")
        assert "{{AutoCoT}}" not in result


class TestProbabilityWeighting:
    def test_logprob_weights_sum_to_one(self):
        logprobs = {"1": -2.3, "2": -0.1, "3": -1.5, "4": -3.0, "5": -4.0}
        weighted, weights = BaseLLMClient._weighted_score_from_logprobs(logprobs, [1,2,3,4,5])
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert 1.0 <= weighted <= 5.0

    def test_frequency_weighted_math(self):
        texts = ["1", "2", "2", "3", "2", "1"]
        weighted, weights = BaseLLMClient._weighted_score_from_samples(texts, [1,2,3], 1, 3)
        expected = (2*3 + 1*2 + 3*1) / 6
        assert abs(weighted - expected) < 1e-6


class TestGEvalEvaluator:
    def test_evaluate_record_mock(self):
        client = MockLLMClient(seed=42)
        evaluator = GEvalEvaluator(client=client, dimensions=["coherence"])
        async def run():
            await evaluator.setup_cot()
            return await evaluator.evaluate_record("source text", "summary text", "coherence")
        result = asyncio.run(run())
        assert "score" in result
        assert "weighted_score" in result
        assert result["is_mock"] is True

    def test_evaluate_dataset_contains_all_dims(self):
        client = MockLLMClient(seed=42)
        evaluator = GEvalEvaluator(client=client, dimensions=["coherence", "fluency"])
        records = [{
            "doc_id": "d1", "system_id": "M1", "source": "src", "system_output": "sum",
            "reference": "ref",
            "scores": {"coherence": 3.0, "consistency": 4.0, "fluency": 2.0,
                       "relevance": 3.5, "overall": 3.1},
        }]
        async def run():
            return await evaluator.evaluate_dataset(records, limit=1, show_progress=False)
        df = asyncio.run(run())
        assert len(df) > 0
        assert "coherence" in df["dimension"].values
        assert "fluency" in df["dimension"].values

    def test_progress_evaluates_each_record_once(self):
        class CountingMockClient(MockLLMClient):
            calls = 0

            async def _complete_with_logprobs(self, *args, **kwargs):
                self.calls += 1
                return await super()._complete_with_logprobs(*args, **kwargs)

        client = CountingMockClient()
        evaluator = GEvalEvaluator(client=client, dimensions=["coherence"])
        record = {
            "doc_id": "d1",
            "system_id": "M1",
            "source": "src",
            "system_output": "sum",
            "reference": "ref",
            "scores": {},
        }

        async def run():
            return await evaluator.evaluate_dataset([record], show_progress=True)

        df = asyncio.run(run())
        assert client.calls == 1
        assert len(df[df["dimension"] == "coherence"]) == 1


def test_mock_auto_cot_does_not_write_to_prompt_directory(tmp_path, monkeypatch):
    import src.geval_evaluator as evaluator_module

    monkeypatch.setattr(evaluator_module, "AUTO_COT_DIR", tmp_path)
    client = MockLLMClient()

    async def run():
        return await generate_auto_cot("coherence", client, force_regenerate=True)

    steps = asyncio.run(run())
    assert steps
    assert not (tmp_path / "auto_cot_coherence.txt").exists()


def test_openai_missing_logprobs_uses_twenty_sample_fallback():
    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "test-model"
    client.provider = "openai"
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="3"),
                logprobs=None,
            )
        ],
        usage=None,
    )

    async def create(**kwargs):
        return response

    async def rate_limited_call(operation):
        return await operation()

    sample_counts = []

    async def sample_n(prompt, n=20, **kwargs):
        sample_counts.append(n)
        return LLMResponse(text="3", score=3, weighted_score=3, provider="openai")

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    client._rate_limited_call = rate_limited_call
    client._sample_n = sample_n

    result = asyncio.run(
        client._complete_with_logprobs("prompt", allowed_scores=[1, 2, 3, 4, 5])
    )
    assert sample_counts == [20]
    assert result.weighted_score == 3


def test_rate_limiter_separates_provider_requests():
    client = MockLLMClient(requests_per_minute=600)
    started = []

    async def operation():
        started.append(time.monotonic())

    async def run():
        await asyncio.gather(
            client._rate_limited_call(operation),
            client._rate_limited_call(operation),
        )

    asyncio.run(run())
    assert started[1] - started[0] >= 0.08
