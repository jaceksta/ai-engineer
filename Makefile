# Thin wrappers around uv. Run `make setup` first on a fresh clone.
.PHONY: setup test lab lint data clean

setup:  ## Create the environment and install pre-commit hooks
	uv sync
	uv run pre-commit install
	@echo "Environment ready. See SETUP.md for the per-machine git identity + SSH steps."

test:  ## Run the test suite (expect NotImplementedError until you implement the exercises)
	uv run pytest

lab:  ## Launch JupyterLab
	uv run jupyter lab

lint:  ## Check formatting and lint rules
	uv run ruff check .
	uv run ruff format --check .

data:  ## Download the default StatsBomb subset into data/raw/
	uv run python data/download_statsbomb.py

clean:  ## Remove caches and generated artifacts (keeps downloaded data)
	rm -rf .pytest_cache .ruff_cache **/__pycache__ mlruns checkpoints
	find . -name '*.pyc' -delete
