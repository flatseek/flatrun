PY ?= python3
SRC := src

.PHONY: help install install-dev test test-fast test-dequant lint format type-check clean build all

help:
	@echo "FlatRun make targets"
	@echo "  make install       install flatrun in editable mode"
	@echo "  make install-dev   install flatrun + dev dependencies"
	@echo "  make test          run the full test suite"
	@echo "  make test-fast     run the test suite, stopping at the first failure"
	@echo "  make test-dequant  run the dequant tests only"
	@echo "  make lint          run ruff"
	@echo "  make format        run ruff --fix"
	@echo "  make type-check    run mypy"
	@echo "  make build         build sdist + wheel"
	@echo "  make clean         remove build artifacts"

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[dev]"

test:
	PYTHONPATH=$(SRC) $(PY) -m pytest src/flatrun/tests -q

test-fast:
	PYTHONPATH=$(SRC) $(PY) -m pytest src/flatrun/tests -x -q

test-dequant:
	PYTHONPATH=$(SRC) $(PY) -m pytest src/flatrun/tests/test_dequant.py -v

lint:
	$(PY) -m ruff check src

format:
	$(PY) -m ruff check --fix src

type-check:
	$(PY) -m mypy src/flatrun

build:
	$(PY) -m pip install build
	$(PY) -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find $(SRC) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: install-dev lint test
