.PHONY: help create-env update-env install install-dev test lint format clean

help:
	@echo "Available commands:"
	@echo "  make create-env    - Create conda environment"
	@echo "  make update-env    - Update conda environment"
	@echo "  make install       - Install package in development mode"
	@echo "  make install-dev   - Install with development dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean up generated files"

create-env:
	conda env create -f environment.yml
	@echo "✓ Environment created. Run: conda activate kaggle-agent"

create-env-dev:
	conda env create -f environment-dev.yml
	@echo "✓ Development environment created. Run: conda activate kaggle-agent-dev"

update-env:
	conda env update -f environment.yml --prune
	@echo "✓ Environment updated"

install:
	pip install -e .
	@echo "✓ Package installed in development mode"

install-dev:
	pip install -e ".[dev]"
	pre-commit install
	@echo "✓ Development dependencies installed"

test:
	pytest tests/ -v --cov=kaggle_agent --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

lint:
	flake8 kaggle_agent/
	mypy kaggle_agent/
	bandit -r kaggle_agent/
	safety check

format:
	black kaggle_agent/ tests/
	isort kaggle_agent/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	rm -rf test_kaggle_project/ test_titanic_demo/ test_*/

check-all: lint test
	@echo "✓ All checks passed!"

conda-export:
	conda env export > environment-freeze.yml
	@echo "✓ Environment exported to environment-freeze.yml"

run-example:
	kaggle-agent init example-titanic --competition titanic
	cd example-titanic && kaggle-agent run --full-auto