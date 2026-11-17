ifeq ($(OS),Windows_NT)
BOOTSTRAP_PYTHON ?= py -3.12
VENV_PYTHON = .venv/Scripts/python.exe
else
BOOTSTRAP_PYTHON ?= python3.12
VENV_PYTHON = .venv/bin/python
endif

PYTHON ?= $(VENV_PYTHON)

.PHONY: install test smoke smoke-real fast phase0 phase1 phase2 phase3 phase4 phase5 all full dashboard

install:
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

phase0: test
	$(PYTHON) -c "from src.utils import load_summeval, print_dataset_stats; print_dataset_stats(load_summeval())"

phase1:
	$(PYTHON) experiments/run_exp_b.py

phase2:
	$(PYTHON) experiments/run_exp_a.py --limit 20 --mock

phase3:
	$(PYTHON) experiments/run_exp_c.py --limit 20 --mock
	$(PYTHON) experiments/run_exp_e.py --limit 80 --mock

phase4:
	$(PYTHON) experiments/run_exp_f.py --limit 80 --mock
	$(PYTHON) experiments/run_exp_g.py --limit 80 --mock
	$(PYTHON) experiments/run_exp_d.py --mock

phase5: dashboard

smoke:
	$(PYTHON) experiments/run_all.py --limit 80 --mock

smoke-real:
	$(PYTHON) experiments/run_exp_a.py --limit 20 --results-dir results/_smoke_real
	$(PYTHON) experiments/run_exp_b.py --limit 20 --results-dir results/_smoke_real

fast:
	$(PYTHON) -m experiments.run_fast

all:
	$(PYTHON) experiments/run_all.py --limit 20

full:
	$(PYTHON) experiments/run_all.py

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py
