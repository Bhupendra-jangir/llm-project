# Architectural and Methodological Decisions

This document records the design choices, architectural decisions, and intentional deviations from the original G-Eval paper (*Liu et al., EMNLP 2023*) implemented in this project.

---

## 1. Mock Client Architecture (`MockLLMClient`)
- **Context**: The pipeline must run without API keys for engineering verification.
- **Decision**: `MockLLMClient` produces deterministic pseudo-random scores and synthetic logprobs, seeded from its request identity. Mock scores are not empirical and must never support research conclusions.
- **Isolation Policy**: LLM-derived mock artifacts are saved under `results/_mock/` and carry `MOCK DATA` labels. Exp B is an API-independent real metric computation, including when invoked as part of a mock pipeline; its output is labeled `REAL DATA`.

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
  - Dimension prompts and their scoring criteria live in `prompts/`; fluency uses the required 1-3 G-Eval scale and the other dimensions use 1-5.
  - Real runs generate and persist Auto-CoT steps when the prompt cache is empty. Mock runs use deterministic local steps and do not write to prompt files.
  - The configured ablations are `full`, `no_cot`, and `no_description`; the project does not claim to implement a separate `no_probs` ablation.
  - The original checkout's upstream reference directory was empty, so prompt provenance is not independently checksum-verified.

---

## 4. Scoring Scale: Evaluator Fluency vs. SummEval Expert Fluency
- **Context**: The G-Eval fluency prompt specifies a 1-3 scale. Inspection of the checked-in SummEval JSON shows that its aggregated expert fluency scores span 1-5, like its other human dimensions.
- **Decision**: Keep G-Eval fluency generation and parsing on 1-3; validate the dataset's human fluency scores on 1-5. Do not rescale or fabricate human labels. Correlations compare the resulting score orderings.

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
  95% confidence intervals for summary-level correlations use 1,000 seeded document-cluster bootstrap resamples. Other reported correlation levels are point estimates.

---

## 7. BERTScore Model Selection
- **Context**: `roberta-large` is the gold standard for BERTScore in academic benchmarks, but running it on CPU-only machines with limited RAM can cause out-of-memory errors.
- **Decision**: `roberta-large` is preferred when supported; CPU-only runs use the configured `distilbert-base-uncased` fallback. The selected model is recorded with each metric row.

---

## 8. Caching and Resumability
- **Context**: Evaluating 1,600 records across 4 dimensions with multiple ablations requires over 19,200 LLM calls. An unexpected network interruption or rate limit could waste time and money.
- **Decision**: A persistent SQLite disk cache uses SHA-256 keys over provider, model, prompt and request parameters. G-Eval score rows are also persisted incrementally for record-level resume. Cache files are local and git-ignored.

---

## 9. Data Leakage Prevention in Learned Combiner
- **Context**: In Experiment G (Unified Score), a supervised Ridge regression model is trained on a subset to learn optimal metric weights.
- **Decision**: Implemented 5-fold `GroupKFold` cross-validation grouped strictly by `doc_id`. This guarantees that summaries generated for the same source document never appear in both train and validation splits, preventing overfitting to specific source document characteristics.

The grid search and learned combiner scale features from each training fold only.
The selected grid weights are then applied to out-of-fold predictions for
comparison. Human scores are targets for supervised validation and are not
components of the automatic unified score.

## 10. Human ratings and bias metadata

Experiment D never generates or imputes human labels. Without
`data/human_ratings.csv`, it reports SummEval expert annotations as the only
human reference and exports a blank rating sheet.

Self-preference analysis is only meaningful when judge-family labels are
configured. Position/order effects are reported as not estimable because the
dataset contains no randomized presentation-order metadata.
