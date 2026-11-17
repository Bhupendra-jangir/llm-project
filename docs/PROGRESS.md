# Project Progress

Status is updated at each phase commit. Generated run outputs are local and
excluded from Git; they are reproducible from the committed code and config.

| Phase | Deliverable | Status | Verification |
|---|---|---|---|
| 0 | Root scaffold, config, data normalization/validation, portable Makefile, baseline tests | Complete | `make test`: 48 passed; `load_summeval()` returned 1,600 records, 100 documents, 16 systems; Ruff passed. |
| 1 | Traditional ROUGE, BLEU, reference/source BERTScore, Exp B correlations | In progress | Focused metric tests pass; full BERTScore run pending. |
| 2 | LLM adapters, G-Eval, caching/resume, mock smoke | Pending | Not yet run. |
| 3 | Experiments A, C, E and correlation figures | Pending | Not yet run. |
| 4 | Experiments D, F, G | Pending | Not yet run. |
| 5 | Dashboard, master runner, methodology and final report | Pending | Not yet run. |

## Phase 0 Notes

- The original project lived under `llm-evaluation-project/`; it is now at
  repository root to match the required commands and layout.
- The two original dataset files were byte-identical. One canonical copy remains
  at `data/summeval.json`.
- SummEval expert fluency scores in this file are on a 1-5 scale. The G-Eval
  fluency prompt/output scale remains 1-3 as required by the paper.
- Phase 0 tests cover scipy correlation agreement, schema and cardinality,
  normalization, bootstrap validation, and document-group split separation.
