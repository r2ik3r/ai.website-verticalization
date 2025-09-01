# Makefile for IAB Verticalizer

PYTHON := python
POETRY := poetry
SRC_DIR := ./src
TESTS_DIR := ./tests

# Tools (executed via Poetry)
RUFF := $(POETRY) run ruff
PYTEST := $(POETRY) run pytest
MYPY := $(POETRY) run mypy

# Canonical run configuration (edit as needed)
DATA_DIR := data
MODELS_DIR := models
OUT_DIR := out
IAB_VERSION := v3
GEO := US
VERSION := 2025-09

# Inputs/Outputs
KAGGLE_CSV ?= /data/kaggle_iab.csv
LABELED_CSV := $(DATA_DIR)/labeled.csv
SITES_URLS := $(DATA_DIR)/sites_urls.csv
PREDS_JSONL := $(OUT_DIR)/preds.jsonl

MODEL_PATH := $(MODELS_DIR)/$(GEO)/$(VERSION)/model.keras
CALIB_PATH := $(MODELS_DIR)/$(GEO)/$(VERSION)/calib.pkl

HELP_COLOR := \033[36m
NO_COLOR := \033[0m

.PHONY: help
help:
	@echo ""
	@echo "Available targets:"
	@echo "  $(HELP_COLOR)install$(NO_COLOR)       Install project (Poetry)"
	@echo "  $(HELP_COLOR)fmt$(NO_COLOR)           Auto-format (ruff format)"
	@echo "  $(HELP_COLOR)lint$(NO_COLOR)          Lint code (ruff check)"
	@echo "  $(HELP_COLOR)test$(NO_COLOR)          Run unit tests (pytest)"
	@echo "  $(HELP_COLOR)typecheck$(NO_COLOR)     Static type checking (mypy)"
	@echo "  $(HELP_COLOR)clean$(NO_COLOR)         Clean caches and build artifacts"
	@echo "  $(HELP_COLOR)run-ingest$(NO_COLOR)    Create labeled CSV from Kaggle input"
	@echo "  $(HELP_COLOR)run-train$(NO_COLOR)     Train model and save artifacts"
	@echo "  $(HELP_COLOR)run-crawl$(NO_COLOR)     Crawl site URLs for inference (optional)"
	@echo "  $(HELP_COLOR)run-infer$(NO_COLOR)     Predict (single model) to JSONL"
	@echo "  $(HELP_COLOR)run-infer-ensemble$(NO_COLOR) Predict with ensemble + site aggregation"
	@echo "  $(HELP_COLOR)run-all$(NO_COLOR)       Ingest → Train → Crawl → Infer"
	@echo ""

# ---------------- Dev basics ----------------

.PHONY: install
install:
	$(POETRY) install --no-interaction --sync

.PHONY: fmt
fmt:
	$(RUFF) format $(SRC_DIR) $(TESTS_DIR)

.PHONY: lint
lint:
	$(RUFF) check $(SRC_DIR) $(TESTS_DIR)

.PHONY: test
test:
	$(PYTEST) -q $(TESTS_DIR)

.PHONY: typecheck
typecheck:
	$(MYPY) $(SRC_DIR)

.PHONY: clean
clean:
	@echo "Cleaning caches and artifacts..."
	rm -rf .ruff_cache .pytest_cache .mypy_cache
	rm -rf **/__pycache__ __pycache__
	rm -rf build dist *.egg-info
	rm -f .coverage coverage.xml

# ---------------- Project run targets ----------------

# 1) Ingest Kaggle IAB dataset → labeled CSV (website,iablabels,contenttext)
.PHONY: run-ingest
run-ingest:
	$(PYTHON) -m src.verticalizer.scripts.ingest_kaggle_iab \
		--kaggle-csv $(KAGGLE_CSV) \
		--out-csv $(LABELED_CSV) \
		--iab-version $(IAB_VERSION)

# 2) Train and save artifacts (model.keras, calib.pkl)
.PHONY: run-train
run-train:
	$(POETRY) run verticalizer train \
		--geo $(GEO) \
		--in $(LABELED_CSV) \
		--version $(VERSION) \
		--out-base $(MODELS_DIR) \
		--epochs 15 \
		--batch-size 64 \
		--labels-loss focal \
		--early-stop \
		--iab-version $(IAB_VERSION)

# 3) Crawl multi-URL pages (optional) for better site-level inference
.PHONY: run-crawl
run-crawl:
	$(POETRY) run verticalizer crawl \
		--urls-csv $(SITES_URLS) \
		--store-html

# 4a) Infer (single model)
.PHONY: run-infer
run-infer:
	$(POETRY) run verticalizer infer \
		--in $(SITES_URLS) \
		--model $(MODEL_PATH) \
		--calib $(CALIB_PATH) \
		--out $(PREDS_JSONL) \
		--iab-version $(IAB_VERSION) \
		--topk 5 \
		--hierarchy-consistent

# 4b) Infer (ensemble, site-level aggregation)
.PHONY: run-infer-ensemble
run-infer-ensemble:
	$(POETRY) run verticalizer infer \
		--in $(SITES_URLS) \
		--models $(MODEL_PATH) $(MODELS_DIR)/$(GEO)/$(VERSION)_b/model.keras \
		--calibs $(CALIB_PATH) $(MODELS_DIR)/$(GEO)/$(VERSION)_b/calib.pkl \
		--out $(PREDS_JSONL) \
		--group-col website \
		--url-col url \
		--page-agg softmax_mean \
		--hierarchy-consistent \
		--iab-version $(IAB_VERSION) \
		--topk 5

# Convenience: End-to-end happy path
.PHONY: run-all
run-all: run-ingest run-train run-crawl run-infer
