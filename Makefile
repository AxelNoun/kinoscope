.PHONY: help setup data embeddings train eval serve test lint typecheck check docker clean

PYTHON ?= python3
EXPERIMENT ?= main

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv, install deps and git hooks
	uv venv --python 3.11 && uv pip install -e ".[serve,dev]"
	pre-commit install && pre-commit install --hook-type commit-msg

data:  ## Run the full DVC data pipeline (ingest -> standardize -> split)
	dvc repro build_dataset

embeddings:  ## Precompute and cache ESM-2 pocket embeddings into LMDB
	$(PYTHON) scripts/precompute_esm_embeddings.py --config-name=config

train:  ## Train a model (override with EXPERIMENT=<name>)
	$(PYTHON) scripts/train.py experiment=$(EXPERIMENT)

eval:  ## Evaluate the last checkpoint on all four split regimes
	$(PYTHON) scripts/evaluate.py --all-splits --bootstrap 1000

serve:  ## Start the inference API locally
	uvicorn kinoscope.serving.api:app --host 0.0.0.0 --port 8000 --reload

test:  ## Run the fast test suite
	pytest -n auto -m "not slow and not network"

test-all:  ## Run every test, including slow and networked ones
	pytest -n auto

lint:
	ruff check src tests scripts && ruff format --check src tests scripts

typecheck:
	mypy src

check: lint typecheck test  ## Full gate: this MUST pass before ticking any MASTER_PLAN box

docker:
	docker build -f docker/Dockerfile --target serve -t kinoscope:latest .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
