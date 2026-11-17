# Project Progress

Status is updated at each phase commit. Generated run outputs are local and
excluded from Git; they are reproducible from the committed code and config.

| Phase | Deliverable | Status | Verification |
|---|---|---|---|
| 0 | Root scaffold, config, data normalization/validation, portable Makefile, baseline tests | Complete | 62 tests pass after subsequent phases; dataset validation confirms 1,600 records, 100 documents, and 16 systems. |
| 1 | Traditional ROUGE, BLEU, reference/source BERTScore, Exp B correlations | Complete | Full Exp B ran from the cached BERTScore results on CPU (`distilbert-base-uncased`), generated a 1,600-row metric table and 65 metric/dimension correlation rows; 95% document-bootstrap CIs use 1,000 resamples. |
| 2 | LLM adapters, G-Eval, caching/resume, mock smoke | In progress | Focused G-Eval tests pass; full mock pipeline verification remains. |
| 3 | Experiments A, C, E and correlation figures | In progress | Experiment code is implemented; end-to-end mock verification remains. |
| 4 | Experiments D, F, G | In progress | Human rating export/import and analysis paths are implemented; end-to-end verification remains. |
| 5 | Dashboard, master runner, methodology and final report | In progress | Dashboard page structure updated; startup verification and evidence-based report remain. |

## Phase 0 Notes

- The original project lived under `llm-evaluation-project/`; it is now at
  repository root to match the required commands and layout.
- The two original dataset files were byte-identical. One canonical copy remains
  at `data/summeval.json`.
- SummEval expert fluency scores in this file are on a 1-5 scale. The G-Eval
  fluency prompt/output scale remains 1-3 as required by the paper.
- Phase 0 tests cover scipy correlation agreement, schema and cardinality,
  normalization, bootstrap validation, and document-group split separation.
- The full traditional-metric correlation run is real dataset output (not LLM
  mock data); its CPU BERTScore implementation uses the configured fallback
  encoder. No real LLM judge run has been performed because no API key is set.
