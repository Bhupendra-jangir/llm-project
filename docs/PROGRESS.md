# Project Progress

Status is updated at each phase commit. Generated run outputs are local and
excluded from Git; they are reproducible from the committed code and config.

| Phase | Deliverable | Status | Verification |
|---|---|---|---|
| 0 | Root scaffold, config, data normalization/validation, portable Makefile, baseline tests | Complete | 62 tests pass after subsequent phases; dataset validation confirms 1,600 records, 100 documents, and 16 systems. |
| 1 | Traditional ROUGE, BLEU, reference/source BERTScore, Exp B correlations | Complete | Full Exp B ran from the cached BERTScore results on CPU (`distilbert-base-uncased`), generated a 1,600-row metric table and 65 metric/dimension correlation rows; 95% document-bootstrap CIs use 1,000 resamples. |
| 2 | LLM adapters, G-Eval, caching/resume, mock smoke | Complete | `run_exp_a.py --limit 20 --mock` wrote all four dimensions and three configured ablations under `results/_mock/`; no prompt files were changed. |
| 3 | Experiments A, C, E and correlation figures | Complete | Mock correlation and ranking outputs verified by `make smoke`; Experiment A reports all three correlation levels and summary-level bootstrap CIs. |
| 4 | Experiments D, F, G | Complete | Mock pipeline verified human alignment, all five bias-analysis outputs (position marked not estimable), grouped grid search and Ridge CV. |
| 5 | Dashboard, master runner, methodology and final report | Complete | AppTest loaded all seven pages without exceptions; `make test` passed 62 tests, Ruff passed, and both `make smoke` and no-key `make smoke-real` completed. |

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
- The final local validation ran `make test` (62 passed), repository-wide Ruff,
  `make smoke` (complete mock pipeline), `make smoke-real` (explicit no-key
  fallback), and Streamlit AppTest over all seven pages. No real LLM evaluation
  was performed; see `FINAL_REPORT.md` for the evidence-based conclusion.
