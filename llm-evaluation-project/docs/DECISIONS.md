# Architectural and Methodological Decisions

This document records the design choices, architectural decisions, and intentional deviations from the original G-Eval paper (*Liu et al., EMNLP 2023*) implemented in this project.

---

## 1. Mock Client Architecture (`MockLLMClient`)
- **Context**: The evaluation pipeline must be completely testable, verifiable, and runnable end-to-end without requiring active paid API keys.
- **Decision**: Implemented `MockLLMClient` in `src/llm_clients.py`. It uses a deterministic hash (SHA-256 of the prompt text combined with an optional random seed) to generate reproducible, realistic scores, synthetic logprob distributions, and token completions.
- **Isolation Policy**: All mock outputs are written exclusively to `results/_mock/` and are never saved to `results/`. Any figure or summary table generated with mock data explicitly features `[MOCK DATA]` in its title and annotations to prevent confusion with empirical research findings.

---

## 2. Unified Evaluation Score Formulation (Experiment G)
- **Context**: The initial concept proposed combining G-Eval, traditional metrics, and a "HumanAlignment" term into a single composite score. However, directly incorporating human alignment would require ground-truth human annotations at inference time, defeating the purpose of an automatic metric and causing severe target leakage.
- **Decision**: Replaced "HumanAlignment" with **LLM-Judge Ensemble Agreement**.
  $$\text{Unified Score} = w_1 \cdot \text{G-Eval} + w_2 \cdot \text{BERTScore} + w_3 \cdot \text{ROUGE} + w_4 \cdot \text{EnsembleAgreement}$$
  Ensemble agreement is computed as the inverted standard deviation of scores across distinct judge models:
  $$\text{Agreement} = \text{Normalize}(1.0 - \sigma_{\text{judges}})$$
  This provides a fully unsupervised, automatic confidence signal: summaries where diverse LLM judges agree receive higher confidence weighting.

---

## 3. Prompt Template Fidelity and Auto-CoT
- **Context**: The paper evaluates four dimensions on SummEval: Coherence, Consistency, Fluency, and Relevance.
- **Decision**:
  - The base prompts from the official reference repository (`reference/geval/prompts/summeval/`) are preserved verbatim in `prompts/`.
  - In the paper, chain-of-thought (CoT) evaluation steps were generated once by GPT-4 and cached. We implement `generate_auto_cot` in `src/geval_evaluator.py`, which generates steps if not present and caches them in `prompts/auto_cot_<dimension>.txt`.
  - Ablation prompts (`no_cot`, `no_description`) are isolated in `prompts/ablation/`.

---

## 4. Scoring Scale: Fluency (1-3) vs. Other Dimensions (1-5)
- **Context**: In SummEval and the G-Eval paper, Fluency was annotated by human experts on a 3-point Likert scale (1: Poor, 2: Fair, 3: Good), whereas Coherence, Consistency, and Relevance were evaluated on a 1-5 scale.
- **Decision**: We maintain the exact 1-3 scale for Fluency across all prompt instructions, score parsing regexes, allowed token sets, and logprob probability normalizations. The other three dimensions strictly use 1-5.

---

## 5. Score Aggregation Protocol: Logprobs vs. Sampling Fallback
- **Context**: G-Eval computes continuous expectation scores via token logprobs:
  $$S = \sum_{i \in \text{Allowed}} i \cdot \frac{\exp(\log p(i))}{\sum_{j} \exp(\log p(j))}$$
  However, certain modern APIs (or local Ollama endpoints) may not support logprob retrieval or return logprobs for chat endpoints.
- **Decision**: The framework supports two interchangeable score aggregation modes (`score_mode` in `config.yaml`):
  1. `logprobs`: Primary mode. Extracts log probabilities of score tokens and computes the mathematical expectation.
  2. `sampling`: Fallback mode. Takes $n=20$ independent samples at temperature 1.0, parses integer outputs, and calculates their arithmetic mean, faithfully mirroring Section 3.2 of the paper.

---

## 6. Correlation Protocols (Three Evaluation Levels)
- **Context**: Literature often confuses summary-level, system-level, and sample-level correlations, leading to incomparable metric comparisons.
- **Decision**: We report all three levels clearly:
  1. **Summary-level (Paper Protocol)**: For each of the 100 source articles, compute Spearman $\rho$, Kendall $\tau$, and Pearson $r$ across the 16 candidate summaries, then average across all 100 documents. This matches Table 1 in Liu et al. (2023).
  2. **System-level**: Aggregate scores per system across all documents (16 points total) and correlate system rankings.
  3. **Sample-level (Pooled)**: Directly correlate all 1,600 pairs without grouping.
  Confidence intervals (95%) are computed via 1,000 cluster-bootstrap iterations grouped by document ID.

---

## 7. BERTScore Model Selection
- **Context**: `roberta-large` is the gold standard for BERTScore in academic benchmarks, but running it on CPU-only machines with limited RAM can cause out-of-memory errors.
- **Decision**: Implemented automatic fallback in `src/traditional_metrics.py`. If CUDA is available, `roberta-large` is loaded. If on CPU, it gracefully defaults to `distilbert-base-uncased` unless explicitly overridden via the `BERTSCORE_MODEL` environment variable.

---

## 8. Caching and Resumability
- **Context**: Evaluating 1,600 records across 4 dimensions with multiple ablations requires over 19,200 LLM calls. An unexpected network interruption or rate limit could waste time and money.
- **Decision**: Built a persistent SQLite disk cache (`DiskCache` in `src/utils.py`). Cache keys are SHA-256 hashes of `(provider, model, prompt, temperature, max_tokens, top_p)`. Any pipeline execution checks the cache before dispatching network requests, enabling seamless resumption from interruptions.

---

## 9. Data Leakage Prevention in Learned Combiner
- **Context**: In Experiment G (Unified Score), a supervised Ridge regression model is trained on a subset to learn optimal metric weights.
- **Decision**: Implemented 5-fold `GroupKFold` cross-validation grouped strictly by `doc_id`. This guarantees that summaries generated for the same source document never appear in both train and validation splits, preventing overfitting to specific source document characteristics.
