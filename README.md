# 🤖➿⚙ Website Verticalizer
Production‑grade system to classify websites into IAB categories and generate geo‑specific Premiumness Scores (1–10).

See also
- docs/website_verticalizer_spec.md — for an oveerview of the project.
- docs/website_verticalizer_deep_dive.md - for deep dive into Website Verticalizer
- docs/premiumness_scoring_spec.md - for deep dive into Premiumness Scoring
- CONTRIBUTING.md — contributor workflow and QA gates.

***

## 📌 Overview
Website Verticalizer is a modular ML pipeline that:
- Performs multilabel IAB Tier‑1 classification (Tier‑2 optional as label coverage grows).
- Assigns geo‑specific Premiumness Scores (1–10) per category when labels are available.
- Operates on labeled or unlabeled inputs via crawl → embed → train → infer → eval stages.
- Combines crawled content, semantic embeddings, a two‑head Keras model, and per‑label calibration.

Built for reproducibility, caching, cost controls, and per‑geo artifact versioning.

***

## 🚀 Features
- Multi‑label IAB classification with Keras on semantic embeddings.
- Pluggable embedder with cache‑first behavior, rate limiting, retries, dry‑run, and max‑calls; Gemini client is implemented; other providers can be added via the same interface.
- Robots.txt‑aware crawler with readability extraction; Postgres metadata; optional object storage for raw HTML and vectors.
- Premiumness Score head per vertical, clamped to 1–10 and normalized during training.
- Per‑label isotonic probability calibration (applied at inference on classification head only).
- Decoupled CLIs: crawl, embed, train, infer, eval; plus run‑pipeline orchestrator.
- File‑driven I/O; structured logging; idempotent reruns.

Note: The codebase ships a Gemini embedder; if another provider is desired, add a provider client matching the Gemini interface.

***

## 🏗 Architecture & Workflow
High‑level flow; see docs/spec.md for the detailed diagrams.
- Inputs (CSV/Excel: websites, labels) → Crawl (robots‑aware) → Embed (cache, dedup, batch) → Train (two‑head Keras, per‑label isotonic) → Artifacts (model.keras, calib.pkl) → Infer (predict probs+scores; apply calibration) → Output JSONL.
- Calibration applies to probabilities only; scores are discretized 1..10 at inference.

Mermaid diagram is in the spec to avoid duplication.

***

## 📂 Project Structure
```
repo/
├── pyproject.toml
├── README.md
├── docs/
│   └── spec.md
├── CONTRIBUTING.md
├── .env.example
├── src/
│   └── verticalizer/
│      ├── cli.py
│      ├── apps/{crawler,embedder,trainer,infer,evaluate}/
│      ├── crawl/        # fetcher, parse, robots
│      ├── embeddings/   # provider clients + cache (Gemini implemented)
│      ├── models/       # Keras, calibration, persistence, registry
│      ├── pipeline/     # common helpers, nodes, io
│      ├── storage/      # Postgres & S3 clients and repositories
│      └── utils/        # logging, taxonomy, metrics, seed
└── tests/
```


***

## 📊 Data Contracts
Pointers only; full contracts live in docs/spec.md.

- Labeled CSV: website, iablabels (JSON/CSV), premiumnesslabels (IAB→1..10), optional contenttext, optional geo; IDs normalized to uppercase IAB codes.
- Unlabeled CSV: website with optional contenttext; supports inference or classification‑only training.
- Predictions JSONL: {website, geo, categories[{id,label,prob,score}], generated_at}; model selection via path or registry.

Example rows and Excel→CSV/JSON converter are documented in the spec and CLI help.

***

## ⚙️ Installation
Prerequisites: Python 3.10+, Poetry; Postgres recommended; S3/minio optional for blobs.

Setup
```
poetry install
cp .env.example .env
```
Core env keys (full list in docs/spec.md):
- GEMINI_API_KEY, GEMINI_EMB_MODEL, GEMINI_EMB_DIM, GEMINI_TASK_TYPE, GEMINI_EMB_DRYRUN, GEMINI_EMB_MAX_CALLS, GEMINI_EMB_RATE_LIMIT.
- DATABASE_URL (Postgres DSN), S3 credentials, HTTP_USER_AGENT/HTTPTIMEOUT for crawler.

Provider note: The runtime embedder is Gemini by default; additional providers require adding a client under embeddings/ with the same caching and rate‑limit hooks.

***

## 📜 Commands
Top‑level entrypoint:
```
poetry run verticalizer [command] [options]
```
- Crawl: robots‑aware fetch and persistence.
- Embed: embed latest text with cache and cost controls.
- Train: fit two‑head Keras and save calibrator; records artifacts per geo/version.
- Infer: prepare embeddings, predict, apply calibrator (probs only), write JSONL.
- Eval: compare predictions vs gold; emits JSON report.
- Run pipeline: Excel→CSV/JSON→train→infer→compare orchestration.

CLI switches and examples are in the spec to avoid duplication.

***

## 📈 Ops & Quality Tips
- Keep taxonomy JSONs consistent across stages; pin seeds for reproducibility.
- Maximize embedding cache hits; use QPS and max‑calls to meet budget; DRYRUN for smoke tests.
- Monitor zero‑vector share, stale embeddings vs lasthash, and crawl errors in logs.

***

## 🔒 Compliance
- robots.txt respected; courtesy delay and configurable UA/timeout.
- No PII; secrets via env; idempotent reruns with durable storage for audits.

***

## 🧪 Testing
```
poetry run pytest
```
Linting and style:
```
poetry run ruff check src --fix
```
For PR process and QA gates, see CONTRIBUTING.md.

***

## 📚 Specification
For scope, requirements, persistence contracts, acceptance gates, diagrams, and roadmap, see docs/spec.md.

***

## 🧭 Troubleshooting
- Gemini key or permission errors: set GEMINI_API_KEY; disable GEMINI_EMB_DRYRUN for real vectors.
- Robots denial: provide contenttext in CSV to bypass crawl; then run embed/infer.
- Empty vectors during dry‑run/max‑calls: expected zero‑vectors; lower restrictions for production runs.

***

## 🏷 License
Proprietary — internal use only.

***