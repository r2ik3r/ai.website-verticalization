# Contributing to Website Verticalizer

Thank you for the interest in contributing. Bug reports, fixes, features, docs, and tests are welcome. This guide focuses on contributor workflow and quality gates. Authoritative user/ops docs live in README.md; detailed scope lives in spec.md.

- Project overview, architecture, environment, commands: see README.md
- Full project requirements and acceptance criteria: see spec.md

***

## Development setup

Prerequisites
- Python 3.10–3.11
- Poetry
- Git

Setup
```
git clone <repo-url>
cd <repo-dir>
poetry install
# choose an environment template:
cp .env.dev.template .env        # cost-efficient, real embeddings
# or
cp .env.prod.template .env       # accuracy-first production
# then set required secrets: GEMINI_API_KEY, optional S3_*, DB_DSN
```
Run the CLI
```
poetry run verticalizer --help
```
For day-to-day commands, prefer the Makefile targets described below and in README.md.

***

## Makefile tasks

Common tasks executed via Poetry:
- make install — install dependencies
- make fmt — format code (ruff format)
- make lint — lint code (ruff check)
- make test — run tests (pytest)
- make typecheck — mypy type checking
- make clean — remove caches and build artifacts

Canonical run targets (wrap the CLIs):
- make run-ingest — create labeled CSV from Kaggle IAB input
- make run-train — train model and save artifacts (model.keras, calib.pkl)
- make run-crawl — optional: crawl URLs for inference
- make run-infer — predict (single model) to JSONL
- make run-infer-ensemble — predict with ensemble + site aggregation
- make run-all — end-to-end ingest → train → crawl → infer

Refer to README.md for outputs and variable knobs (GEO, VERSION, IAB_VERSION, paths).

***

## Workflow

1) Create an issue (bug/feature) or pick one from the tracker.
2) Branch naming:
   - feature/<slug>
   - fix/<slug>
   - docs/<slug>
3) Implement the change; keep PRs small and focused.
4) Run the local quality gate:
```
make fmt && make lint && make test && make typecheck
```
5) Open a PR: explain changes, link issues, and include logs/screenshots if CLI UX changed.
6) Address review feedback until approval, then merge.

***

## Style & testing

- Formatter: Ruff (ruff format)
- Linter: Ruff (ruff check)
- Types: mypy (strictness settings in pyproject.toml)
- Tests: pytest under ./tests

Quick commands
```
make fmt
make lint
make test
make typecheck
```
Keep PRs lint-clean. If adding rules, edit pyproject.toml accordingly.

***

## Environment & secrets

- See README.md → Installation for environment variables and templates (env.dev.template, env.prod.template).
- Never commit secrets. .env is gitignored. Use distinct secrets for dev and prod.

***

## Backwards compatibility

If changing CLI flags or I/O formats, update:
- README.md
- Any module-level docs under src/verticalizer/apps/*/ if present
Document any breaking changes clearly in the PR and changelog.

***

## Code areas of interest

- apps/: CLI layers and orchestration per stage (crawl, embed, train, infer, evaluate)
- embeddings/: embedding clients and cache
- models/: Keras heads, calibration, persistence/registry
- pipeline/: training/inference nodes and utilities
- storage/: Postgres and S3/MinIO repositories/clients
- utils/: taxonomy, metrics, logging, seed

***

## Recognition

All contributions are appreciated — fixes, features, docs, and tests. Thanks for helping improve Website Verticalizer.

***