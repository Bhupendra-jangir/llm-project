# Initial Project Audit

This audit records the repository state before implementation work began on
`complete-project`. The original project was checked out under
`llm-evaluation-project/` instead of the repository root; the module status
below refers to that original code.

## Planned modules

| Module | Status | Initial findings |
|---|---|---|
| `configs/config.yaml` | PARTIAL | Most experiment settings existed, but key defaults and result routing did not consistently prevent mock runs from writing into real results. |
| `data/summeval.json` | PARTIAL | Two byte-identical copies existed, one at the repository root and another under the nested project. No complete 1,600-record / 100-document / 16-system validation was enforced by the loader. |
| `prompts/` | PARTIAL | Four dimension prompts and two ablation prompts existed, but the upstream G-Eval reference clone was empty and Auto-CoT/no-description prompt coverage was incomplete. |
| `src/utils.py` | PARTIAL | Loading, conversion, statistics, cache, correlations, and bootstrap helpers existed. Schema validation was shallow; bootstrap exceptions were silently discarded. |
| `src/llm_clients.py` | PARTIAL | Provider adapters, a mock, a semaphore, retry logic, and SQLite caching existed. The cache key omitted some request parameters, some provider logprob fallbacks did not enforce 20 independent samples, and mock/rate-limit isolation needed verification. |
| `src/geval_evaluator.py` | PARTIAL | Core scoring and prompt handling existed, but errors were converted to NaN records, mock Auto-CoT could write prompt files, progress evaluation duplicated work, and runs did not reliably resume at record granularity. |
| `src/traditional_metrics.py` | PARTIAL | ROUGE, BLEU, and BERTScore code existed. Existing metric cache loading ignored requested limits and BERTScore failures could return NaNs while appearing successful. |
| `src/human_alignment.py` | PARTIAL | Stratified subset and rating template existed. Import validation was permissive, ratings were not fully aligned/validated, and Cohen/weighted kappa and Pearson reporting were absent. |
| `src/bias_analysis.py` | PARTIAL | Several residual, length, style, and system-type summaries existed, but position/order effects and significance reporting for all requested analyses were absent. |
| `src/unified_score.py` | BROKEN | Fixed/grid/learned paths existed, but fallback components could make the score misleading, grid search evaluated on the target data, and cross-validation preprocessing was not isolated from folds. |
| `src/generate_responses.py` | PARTIAL | Optional generation was present, but output routing and API-key/cost behavior were not consistently controlled. |
| `experiments/run_exp_a.py` | PARTIAL | G-Eval dimensions and ablations were present, but the results protocol omitted required CI coverage and depended on evaluator behavior that could silently emit NaNs. |
| `experiments/run_exp_b.py` | PARTIAL | Traditional metrics and correlation output existed, but cache/limit handling and reference/source metric reliability were unresolved. |
| `experiments/run_exp_c.py` | PARTIAL | Multiple judges were supported, but rankings versus human system rankings and mock output isolation were incomplete. |
| `experiments/run_exp_d.py` | PARTIAL | Subset/rating export existed, but requested reliability statistics and robust import behavior were missing. |
| `experiments/run_exp_e.py` | PARTIAL | Correlations and plots existed, but output naming, bootstrap completeness, and mock-vs-real labeling were inconsistent. |
| `experiments/run_exp_f.py` | PARTIAL | It invoked existing bias calculations, but did not cover all five required bias analyses or produce complete significance results. |
| `experiments/run_exp_g.py` | BROKEN | It could proceed without the required G-Eval/ensemble inputs and compare an in-sample tuned grid score as if it were held out. |
| `experiments/run_all.py` | BROKEN | Mock mode did not consistently route every experiment to `results/_mock/`; the pipeline could mix mock and real results. |
| `dashboard/app.py` | BROKEN | Seven pages existed, but their names and semantics did not match the requested page set; mock detection could fall back to real data and some pages swallowed errors. |
| `tests/` | PARTIAL | Tests covered selected utilities and metrics, but did not enforce the complete data cardinality, exact scipy comparisons for all correlations, or document-group leakage guarantees. |
| `docs/DECISIONS.md` | PARTIAL | Decisions were documented, but some claims (notably prompt provenance, mock file isolation, and implemented analyses) did not match the checked-in code. |
| `docs/METHODOLOGY.md` | PARTIAL | A methodology draft existed, but its descriptions overstated implementation and did not provide an evidence-based final result. |
| `docs/PROGRESS.md` | BROKEN | It marked most phases complete despite missing or incomplete deliverables and passing verification evidence. |
| `docs/FINAL_REPORT.md` | MISSING | No final report or run-derived results were present. |
| `docs/AUDIT.md` | MISSING | This file records the initial audit. |
| `.env.example` | PARTIAL | Provider variables were listed, but no validation of keys or reliable environment setup was described. |
| `.gitignore` | MISSING | There was no ignore policy for `.env`, virtualenvs, caches, generated results, or model artifacts. |
| `requirements.txt` | PARTIAL | Dependencies were pinned, but no Python environment was present and a number of requirements were unnecessarily heavyweight for no-key metric/smoke runs. |
| `Makefile` | BROKEN | It hard-coded a Windows user path and invoked `python`, which was unavailable in the initial environment. |
| `README.md` | PARTIAL | Setup and usage claims did not match repository placement, mock routing, or the actual verification state. |

## Baseline commands and failures

The initial environment had Python 3.14.6 on `PATH`, but no project dependencies.
Python 3.12.13 was available and was selected for the project virtual
environment.

| Command | Initial result |
|---|---|
| `python3 -m pytest tests/ -v --tb=short` | Failed before collection: `No module named pytest`. |
| `make smoke` | Failed immediately: `make: python: No such file or directory`. |
| `python3.12 experiments/run_exp_b.py --limit 1` | Failed at import: `ModuleNotFoundError: No module named 'numpy'`. |

The provided project report and G-Eval paper PDF were checked. The paper
reports 0.514 summary-level Spearman and 0.418 Kendall for G-Eval-4 on
SummEval. The repository's `reference/geval/` directory was empty, so prompt
provenance could not be established from the local clone.

## Layout correction made before implementation

The tracked project was moved from `llm-evaluation-project/` to the repository
root to match the requested structure. The duplicate nested dataset was
removed after its SHA-256 was confirmed identical to `data/summeval.json`.
The supplied PDFs were moved from the case-colliding `Docs/` path to
`reference/` so the lowercase `docs/` project directory works on
case-insensitive filesystems as well as Linux.
