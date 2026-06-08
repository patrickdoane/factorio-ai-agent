PYTHON ?= python3.11
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: setup test run-scripted run-random train-ppo clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e '.[dev]'

test:
	$(VENV_PYTHON) -m pytest

run-scripted:
	$(VENV)/bin/factorio-ai run-scripted

run-random:
	$(VENV)/bin/factorio-ai run-random

train-ppo:
	$(VENV)/bin/factorio-ai train-ppo

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
