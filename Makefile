PYTHON ?= python3.11
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: setup test run-scripted run-random evaluate train-ppo clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e '.[dev]'

test:
	$(VENV_PYTHON) -m pytest

run-scripted:
	$(VENV)/bin/factorio-ai run-scripted --task three-plates

run-random:
	$(VENV)/bin/factorio-ai run-random

evaluate:
	$(VENV)/bin/factorio-ai evaluate --agent both --task three-plates --episodes 10 --seed 42

train-ppo:
	$(VENV)/bin/factorio-ai train-ppo --task first-plate --total-timesteps 256 --n-steps 64 --batch-size 32 --eval-episodes 3

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
