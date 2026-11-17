# ============================================================
# Makefile – Enhanced LLM Evaluation Framework
# ============================================================

PYTHON = python
PROJECT_DIR = c:/Users/bhupe/Desktop/llm project/llm-evaluation-project
VENV = .venv

.PHONY: setup smoke smoke-real phase0 phase1 phase2 phase3 phase4 phase5 all test clean

# Install all dependencies
setup:
	$(PYTHON) -m pip install -r requirements.txt

# Run the full test suite
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# Smoke test with mock client (no API keys needed)
smoke:
	@echo "=== Running smoke test with MockLLMClient (no API keys needed) ==="
	$(PYTHON) experiments/run_exp_a.py --limit 10 --mock --results-dir results/_mock
	$(PYTHON) experiments/run_exp_b.py --limit 10 --results-dir results/_mock
	$(PYTHON) experiments/run_exp_e.py --limit 10 --mock --results-dir results/_mock
	$(PYTHON) experiments/run_exp_f.py --limit 10 --mock --results-dir results/_mock
	$(PYTHON) experiments/run_exp_g.py --limit 10 --mock --results-dir results/_mock
	@echo "=== Smoke test complete. Results in results/_mock/ ==="

# Smoke test with real API keys (--limit 20)
smoke-real:
	@echo "=== Running smoke test with real API (--limit 20) ==="
	$(PYTHON) experiments/run_exp_a.py --limit 20
	$(PYTHON) experiments/run_exp_b.py --limit 20
	@echo "=== Smoke-real complete. Check results/ ==="

# Phase-by-phase runs
phase0:
	$(PYTHON) -c "from src.utils import load_summeval, print_dataset_stats; d=load_summeval(); print_dataset_stats(d)"
	$(PYTHON) -m pytest tests/ -v --tb=short

phase1:
	$(PYTHON) experiments/run_exp_b.py

phase2:
	@echo "=== G-Eval smoke test (mock, limit=20) ==="
	$(PYTHON) experiments/run_exp_a.py --limit 20 --mock
	@$(PYTHON) -c "\
import os; \
env_ok = os.path.exists('.env') and any(k in open('.env').read() for k in ['OPENAI_API_KEY','GEMINI_API_KEY','ANTHROPIC_API_KEY','GROQ_API_KEY']); \
print('API key found — ready for real run.' if env_ok else '\n*** NO API KEY FOUND ***\nAdd your key to .env (see .env.example) then run: make smoke-real\nFull run: make all\n')"

phase3: phase1
	$(PYTHON) experiments/run_exp_a.py
	$(PYTHON) experiments/run_exp_c.py
	$(PYTHON) experiments/run_exp_e.py

phase4: phase3
	$(PYTHON) experiments/run_exp_f.py
	$(PYTHON) experiments/run_exp_g.py
	$(PYTHON) experiments/run_exp_d.py

phase5:
	$(PYTHON) -m streamlit run dashboard/app.py

# Full pipeline (requires API keys for G-Eval phases)
all: phase1 phase3 phase4
	@echo "=== All experiments complete. Run 'make phase5' to launch the dashboard. ==="

# Remove pycache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
