ifeq ($(OS),Windows_NT)
BOOTSTRAP_PYTHON ?= py -3.12
VENV_PYTHON = .venv/Scripts/python.exe
else
BOOTSTRAP_PYTHON ?= python3.12
VENV_PYTHON = .venv/bin/python
endif

PYTHON ?= $(VENV_PYTHON)

.PHONY: install test smoke smoke-real phase0 phase1 phase2 phase3 phase4 phase5 all dashboard

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
	$(PYTHON) experiments/run_exp_c.py --limit 20
	$(PYTHON) experiments/run_exp_e.py --limit 80

phase4:
	$(PYTHON) experiments/run_exp_f.py --limit 80
	$(PYTHON) experiments/run_exp_g.py --limit 80
	$(PYTHON) experiments/run_exp_d.py

phase5: dashboard

smoke:
	$(PYTHON) experiments/run_all.py --limit 80 --mock

smoke-real:
	$(PYTHON) experiments/run_exp_a.py --limit 20
	$(PYTHON) experiments/run_exp_b.py --limit 20

all:
	$(PYTHON) experiments/run_all.py --full

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py
