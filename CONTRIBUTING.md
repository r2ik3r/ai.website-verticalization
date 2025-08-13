# Contributing to Website Verticalizer

Thank you for your interest in contributing!
We welcome bug reports, fixes, features, documentation, and tests.

To avoid duplication, the authoritative user/ops documentation (architecture, environment, CLI usage) lives in README.md, and the full scope/requirements live in spec.md. This guide focuses on contributor workflow and quality gates.

- Project overview, architecture, environment, commands: see README.md
- Full project requirements and acceptance criteria: see spec.md

---

## Development Setup

Prerequisites
- Python 3.11+
- Poetry
- Git

Setup
```
git clone
cd
poetry install
cp .env.example .env   # set GEMINI_API_KEY, DATABASE_URL, S3_*, GEMINI_EMB_*
# optional:
poetry shell
```

Run the CLI
```
poetry run verticalizer  --help
```

For detailed commands and examples, see the “📜 Commands” section in README.md.

---

## Makefile Tasks

Common tasks (executed via Poetry):

- make install — install dependencies
- make fmt — format code (ruff format)
- make lint — lint code (ruff check)
- make test — run tests (pytest)
- make cov — coverage
- make clean — remove caches and build artifacts
- make qa — fmt + lint + test

---

## Workflow

1) Create an issue (bug/feature) or pick one from the tracker.  
2) Branch:
   - feature/
   - fix/
   - docs/
3) Implement your change; keep PRs small and focused.  
4) Run the quality gate locally:
```
make qa
```
5) Open a PR (describe changes, link issues, attach logs/screenshots if CLI UX changed).  
6) Address review feedback until approval and merge.

---

## Style & Testing

- Formatter: Ruff (ruff format)
- Linter: Ruff (ruff check)
- Tests: pytest under ./tests (or adjust Makefile)

Commands
```
make fmt
make lint
make test
make cov
```

Keep PRs lint-clean. If adding new rules, update pyproject.toml.

---

## Environment & Secrets

- See README.md → Installation for environment variables.
- Never commit secrets. `.env` is gitignored.

---

## Backwards Compatibility

- If you change CLI flags or I/O formats, update:
  - README.md
  - Module-level READMEs under src/verticalizer/apps/*/
- Document any breaking changes clearly in the PR and changelog.

---

## Recognition

We appreciate all contributions — fixes, features, docs, and tests.
Thanks for helping improve IAB Verticalizer!

---