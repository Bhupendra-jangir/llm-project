# Final Research Report

## Research question

Can combining G-Eval with traditional metrics and human-alignment analysis
provide more reliable model rankings than any single automatic metric?

## Run provenance

- Dataset: checked-in SummEval JSON, validated at 1,600 records, 100 documents,
  and 16 systems.
- Traditional metrics: full-dataset Experiment B, run locally with Python 3.12
  and CPU BERTScore fallback `distilbert-base-uncased`.
- LLM judges: no real API keys were configured. G-Eval and judge outputs in
  `results/_mock/` are deterministic smoke-test data and are excluded from all
  research conclusions below.
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

## G-Eval paper comparison

Liu et al. report 0.514 summary-level Spearman for G-Eval-4 on SummEval. A
real G-Eval reproduction was **not run**, so this project has no empirical
value to compare with 0.514 and no defensible distance-from-paper estimate.
The mock smoke-test correlations are deliberately not compared to the paper.
A full real run must first be performed with configured API credentials and
the same dataset protocol.

## Answer to the research question

**Inconclusive.** Experiment B supplies full-dataset traditional baselines, but
there are no real G-Eval, multi-judge, or unified-score predictions to compare
against those baselines. Mock outputs only verify execution and output
contracts. It would be misleading to claim that the multi-metric framework
does or does not outperform individual metrics.

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

## Reproduction

1. Install dependencies with `make install`.
2. Copy `.env.example` to `.env`, add the required provider credentials, and
   set `llm.provider`/`llm.model` in `configs/config.yaml`.
3. Configure at least two real entries in `judge_models`; the checked-in
   entries default to mock models for safe API-free execution.
4. Run `make smoke-real` and `make all` first. Review each cost estimate.
5. Run `make full` for the 1,600-record pipeline. The SQLite request cache
   allows completed requests to be reused.
