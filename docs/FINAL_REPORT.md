# Final Research Report

## Research question

Can combining G-Eval with traditional metrics and human-alignment analysis
provide more reliable model rankings than any single automatic metric?

## Run provenance

- Dataset: checked-in SummEval JSON, validated at 1,600 records, 100 documents,
  and 16 systems.
- Traditional metrics: full-dataset Experiment B, run locally with Python 3.12
  and CPU BERTScore fallback `distilbert-base-uncased`.
- LLM judge: one Groq judge ran the bounded pilot on 20 records from 10
  documents and two systems, using four dimensions, one sample per score, and
  the `no_cot` ablation. The run encountered rate limits but saved scores for
  all 20 records; see `results/fast/run_metadata.json`.
- Human ratings: no new human rating file was present. SummEval's existing
  expert annotations are the only human reference used.

## Real-data result: Experiment B

The table reports the strongest traditional metric for each expert-score
dimension. Correlations are summary-level Spearman values: compute a
correlation across systems within each document, then average across
documents. Confidence intervals are seeded 95% document-cluster bootstrap
intervals with 1,000 resamples.

| Expert dimension | Best metric | Spearman | 95% CI |
|---|---|---:|---:|
| Coherence | BERTScore precision vs. source | 0.1539 | [0.0986, 0.2109] |
| Consistency | BERTScore precision vs. source | 0.3720 | [0.3286, 0.4196] |
| Fluency | BERTScore precision vs. source | 0.2904 | [0.2486, 0.3358] |
| Relevance | BERTScore recall vs. reference | 0.3116 | [0.2469, 0.3735] |
| Overall | BERTScore precision vs. source | 0.2779 | [0.2234, 0.3313] |

These are API-free traditional-metric results; they do not represent G-Eval.
The results show that metric performance varies by dimension, but do not by
themselves establish that a composite metric improves rankings.

## Bounded Groq pilot

The pilot produced 100 score rows across 20 summaries: four judged dimensions
and a derived overall score. No score rows were missing. On this small sample,
the best unified score had summary-level Spearman $\rho=0.1111$, while the
best individual metric (`rouge2_ref`) had $\rho=0.3333$. Thus, this pilot did
not show an advantage for the unified score. The pilot also experienced Groq
rate limits and retries.

These estimates are highly exploratory: only 10 documents and two systems
were included, one judge and one sample were used, and no new human ratings or
multi-judge agreement estimates were collected. They must not be interpreted
as full-dataset or statistically robust findings.

## G-Eval paper comparison

Liu et al. report 0.514 summary-level Spearman for G-Eval-4 on SummEval. The
20-record pilot is not a full reproduction and is not directly comparable to
that benchmark. A full real run must use the complete dataset and the same
evaluation protocol.

## Answer to the research question

**Inconclusive.** The bounded pilot's best unified score did not exceed the
best individual metric, but its small sample is insufficient to answer the
research question. Full-dataset G-Eval and multi-judge comparisons, plus human
validation, are still needed for a defensible conclusion.

## Limitations

1. The CPU BERTScore run uses `distilbert-base-uncased`, not the preferred
   `roberta-large`; the model name is recorded in the metric output.
2. No new human annotations were collected. Experiment D exports a blank
   rating sheet but does not fabricate ratings.
3. Position/order effects are not estimable because SummEval contains no
   randomized presentation-order metadata.
4. Self-preference comparisons require explicit judge-family mappings;
   unmapped judge/system pairs are not interpreted as family matches.
5. Bootstrap confidence intervals are currently reported for summary-level
   correlations; pooled and system-level correlations are point estimates.
6. G-Eval's prompt fidelity has not been independently checksum-verified
   against the upstream repository in this checkout.
7. The Groq pilot used one sample per score and encountered rate limits; retries
   increase request count and the resulting correlations have wide uncertainty.

## Reproduction

1. Install dependencies with `make install`.
2. Copy `.env.example` to `.env` and set `GROQ_API_KEY`.
3. Run `python -m experiments.run_fast --dry-run`, then
   `python -m experiments.run_fast` for the bounded single-judge pilot.
4. For research conclusions, configure the full dataset and multiple real
   judges, review cost estimates, and run the complete pipeline.
