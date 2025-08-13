Makefile for IAB Verticalizer
Variables
PYTHON := python
POETRY := poetry
SRC_DIR := ./src
TESTS_DIR := ./tests

Tools (executed via Poetry)
RUFF := $(POETRY) run ruff
PYTEST := $(POETRY) run pytest
MYPY := $(POETRY) run mypy
SAFETY := $(POETRY) run safety

Help text helper
HELP_COLOR := \033[36m
NO_COLOR := \033[0m

.PHONY: help
help:
@echo ""
@echo "Available targets:"
@echo " $(HELP_COLOR)install$(NO_COLOR) Install project (Poetry) and dev tools"
@echo " $(HELP_COLOR)lock$(NO_COLOR) Update dependency lock (poetry lock)"
@echo " $(HELP_COLOR)fmt$(NO_COLOR) Auto-format (ruff format)"
@echo " $(HELP_COLOR)lint$(NO_COLOR) Lint code (ruff check)"
@echo " $(HELP_COLOR)test$(NO_COLOR) Run unit tests (pytest)"
@echo " $(HELP_COLOR)cov$(NO_COLOR) Run tests with coverage report"
@echo " $(HELP_COLOR)typecheck$(NO_COLOR) Static type checking (mypy) [optional]"
@echo " $(HELP_COLOR)security$(NO_COLOR) Dependency vulnerability scan (safety) [optional]"
@echo " $(HELP_COLOR)clean$(NO_COLOR) Clean caches and build artifacts"
@echo " $(HELP_COLOR)qa$(NO_COLOR) Run fmt, lint, and tests"
@echo " $(HELP_COLOR)all$(NO_COLOR) Run fmt, lint, tests, and coverage"
@echo ""

.PHONY: install
install:
$(POETRY) install --no-interaction --sync

.PHONY: lock
lock:
$(POETRY) lock --no-update

.PHONY: fmt
fmt:
$(RUFF) format $(SRC_DIR) $(TESTS_DIR)

.PHONY: lint
lint:
$(RUFF) check $(SRC_DIR) $(TESTS_DIR)

.PHONY: test
test:
$(PYTEST) -q $(TESTS_DIR)

.PHONY: cov
cov:
$(PYTEST) --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=xml $(TESTS_DIR)

.PHONY: typecheck
typecheck:
$(MYPY) $(SRC_DIR)

.PHONY: security
security:
$(SAFETY) check --full-report || true

.PHONY: clean
clean:
@echo "Cleaning caches and artifacts..."
rm -rf .ruff_cache .pytest_cache .mypy_cache
rm -rf **/pycache pycache
rm -rf build dist *.egg-info
rm -f .coverage coverage.xml

.PHONY: qa
qa: fmt lint test

.PHONY: all
all: fmt lint cov