# Enhanced LLM Evaluation Framework

> **Research question**: "Can a multi-metric LLM evaluation framework, combining G-Eval with traditional metrics and human-alignment analysis, provide more reliable model rankings than any single automatic evaluation metric?"

Extends *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment* (Liu et al., EMNLP 2023) with:
- Multi-metric evaluation (ROUGE, BLEU, BERTScore + G-Eval)
- Multi-model judge comparison
- Bias analysis (length, extractiveness, system-type)
- Unified score with learned combination
- 7-page Streamlit dashboard

---

## Quick Start

Python 3.10–3.12 is supported; the pinned PyTorch stack is validated here
with Python 3.12.

```bash
# 1. Create a virtual environment and install pinned dependencies
make install

# 2. Run the end-to-end mock pipeline (NO API KEY NEEDED)
make smoke

# 3. Launch the dashboard (no API key needed)
streamlit run dashboard/app.py
```

## To Run with Real API Keys

```bash
# 1. Copy and fill in provider API key(s)
cp .env.example .env
# Edit .env; never commit this file.

# 2. Update configs/config.yaml:
#    - set llm.provider and llm.model for the baseline judge
#    - replace default mock judge_models with at least two real judges

# 3. Cost-limited real smoke test (20 records)
make smoke-real

# 4. Cost-limited complete pipeline (20 records)
make all

# 5. Review the printed cost estimates, then run all 1,600 records
make full

# 6. Launch dashboard
streamlit run dashboard/app.py
```

With no API key configured, G-Eval and judge experiments explain how to
configure a real run and use mock mode only. `make full` requires real
credentials for the primary judge and all configured judges.

## Current Results (Real, API-Free Metrics)

Experiment B completed on all 1,600 SummEval records using the CPU fallback
`distilbert-base-uncased`. Best summary-level Spearman by expert dimension:

| Expert dimension | Best metric | Spearman | 95% document-bootstrap CI |
|---|---|---:|---:|
| Coherence | BERTScore precision vs. source | 0.154 | [0.099, 0.211] |
| Consistency | BERTScore precision vs. source | 0.372 | [0.329, 0.420] |
| Fluency | BERTScore precision vs. source | 0.290 | [0.249, 0.336] |
| Relevance | BERTScore recall vs. reference | 0.312 | [0.247, 0.374] |
| Overall | BERTScore precision vs. source | 0.278 | [0.223, 0.331] |

These are real traditional-metric correlations, not G-Eval results. No real
LLM-judge run has been performed; mock scores are strictly for smoke tests and
are not research evidence. The research question remains inconclusive pending
a real, full-dataset judge run.
---

## Project Structure

```
llm-project/
├── configs/config.yaml          # All settings (provider, model, limits)
├── data/
│   ├── summeval.json            # SummEval dataset (1600 records)
│   ├── cache/llm_cache.db       # SQLite API cache (auto-created)
│   └── human_ratings.csv        # Optional: your manual ratings
├── prompts/
│   ├── coherence.txt            # G-Eval dimension prompts
│   ├── consistency.txt
│   ├── fluency.txt
│   ├── relevance.txt
│   ├── auto_cot_*.txt           # Auto-CoT prompt cache files
│   └── ablation/                # Ablation variant prompts
├── src/
│   ├── utils.py                 # Data loading, caching, correlations, bootstrap
│   ├── llm_clients.py           # LLM adapters (Mock, OpenAI, Gemini, Anthropic, ...)
│   ├── geval_evaluator.py       # G-Eval core + ablations
│   ├── traditional_metrics.py   # ROUGE, BLEU, BERTScore
│   ├── human_alignment.py       # Subset, IRR
│   ├── bias_analysis.py         # 5 bias analyses
│   ├── unified_score.py         # Weighted combination + learned combiner
│   └── generate_responses.py    # Optional fresh summary generation
├── experiments/
│   ├── run_exp_a.py             # G-Eval baseline (Exp A)
│   ├── run_exp_b.py             # Traditional metrics (Exp B) — NO API NEEDED
│   ├── run_exp_c.py             # Multi-model comparison (Exp C)
│   ├── run_exp_d.py             # Human validation (Exp D)
│   ├── run_exp_e.py             # Correlation matrix (Exp E)
│   ├── run_exp_f.py             # Bias analysis (Exp F)
│   ├── run_exp_g.py             # Unified score (Exp G)
│   └── run_all.py               # Master runner
├── results/                     # All CSV + figure outputs
│   └── _mock/                   # Mock judge outputs (clearly labeled)
├── dashboard/app.py             # 7-page Streamlit dashboard
├── tests/                       # pytest test suite
└── docs/
    ├── DECISIONS.md             # Design decisions + paper deviations
    ├── METHODOLOGY.md           # Methodology documentation
    ├── PROGRESS.md              # Phase completion log
    └── FINAL_REPORT.md          # Full research report
```

---

## Supported LLM Providers

| Provider | Env Variable | Config `provider` | Logprobs |
|----------|-------------|-------------------|----------|
| MockLLMClient | (none needed) | `mock` | ✅ (fake) |
| OpenAI | `OPENAI_API_KEY` | `openai` | ✅ |
| Google Gemini | `GEMINI_API_KEY` | `gemini` | ❌ (falls back to sampling) |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic` | ❌ (falls back to sampling) |
| Groq | `GROQ_API_KEY` | `groq` | ✅ |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter` | ✅ |
| Ollama (local) | `OLLAMA_HOST` | `ollama` | ❌ (falls back to sampling) |

---

## Key CLI Flags

All experiment scripts support:
```
--limit N        Subsample to first N records (for testing)
--mock           Force MockLLMClient (overrides config)
--results-dir    Custom output directory
--seed           Random seed (default: 42)
```

---

## Tests

```bash
pytest tests/ -v --tb=short
```

Covers: score parsing, correlations (validated against scipy), mock client determinism,
probability weighting math, unified score no-leakage, data loading.

---

## References

- Liu et al. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*. EMNLP 2023.
- Fabbri et al. (2021). *SummEval: Re-evaluating Summarization Evaluation*. TACL.
- Zhang et al. (2020). *BERTScore: Evaluating Text Generation with BERT*.
- Post (2018). *A Call for Clarity in Reporting BLEU Scores* (sacrebleu).
