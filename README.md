# 📊 IAB Verticalizer
Production-grade classification system for website content into IAB categories with geo-specific Premiumness Scores

See also:
- spec.md (Project Specification: scope, requirements, acceptance)
- CONTRIBUTING.md (Contributor workflow, QA gates)

---

## 📌 Overview

IAB Verticalizer is a production-ready ML system for:
- Classifying websites into IAB categories (Tier‑1; optionally Tier‑2).
- Assigning geo-specific Premiumness Scores (1–10) per category.
- Operating on labeled datasets, unlabeled datasets (via crawling, embedding, inference), or both in hybrid loops.
- Combining crawled website content + Gemini embeddings + Keras classifier with calibration.
- Achieving production-level robustness, modularity, caching, and reproducibility.

Inspired by “Kaggle Day 2 — Classifying embeddings with Keras,” extended for multi‑label, multi‑geo, production deployments.

---

## 🚀 Features

- Multi‑label classification with Keras and semantic embeddings  
- Gemini API integration with caching, rate limiting, and dry‑run modes  
- Website crawling with robots.txt compliance  
- Geo‑specific Premiumness Scores (1–10)  
- Probability calibration (Isotonic Regression)  
- Modular pipeline: crawl, embed, train, infer, eval — each runnable independently  
- File‑driven I/O with Postgres & optional S3  
- Logging, retries & idempotent reruns for production

---

## 🏗 Architecture & Workflow

```mermaid
flowchart LR
subgraph Input
A[CSV: websites, labels]
end

subgraph Apps
CRAWL[Crawlerapps/crawler]
EMBED[Embedderapps/embedder]
TRAIN[Trainerapps/trainer]
INFER[Inferapps/infer]
EVAL[Evaluateapps/evaluate]
end

subgraph Pipeline
COMMON["prepare_embeddings_for_df()pipeline/common.py"]
NODES["nodes.py(train, infer, eval)"]
end

subgraph Storage
PG[(Postgres)]
S3[(Object Store)]
end

subgraph Artifacts
MODELS[(Model + Calib Registry)]
PRED[Predictions JSONL]
REPORT[Eval JSON]
end

%% Input feeds site lists/labels to apps
A --> CRAWL
A --> EMBED
A --> TRAIN
A --> INFER
A --> EVAL

%% Crawler persists content
CRAWL --> PG
CRAWL --> S3

%% Embedder reads latest crawls and persists embeddings metadata
PG --> EMBED
S3 --> EMBED
EMBED --> PG

%% Shared prep pulls text/embeddings via storage
PG --> COMMON
S3 --> COMMON
COMMON --> NODES

%% Training uses common prep and writes models
TRAIN --> NODES
NODES --> PG
NODES --> MODELS

%% Inference uses models + common prep and writes predictions
MODELS --> INFER
INFER --> NODES
NODES --> PRED

%% Evaluation reads predictions (or model + gold) and writes report
PRED --> EVAL
MODELS --> EVAL
EVAL --> REPORT
```

---

## 📂 Project Structure

```
verticalizer/
├── pyproject.toml
├── README.md
├── spec.md
├── CONTRIBUTING.md
├── .env.example
├── src/verticalizer/
│   ├── cli.py                     # top-level CLI
│   ├── apps/
│   │   ├── crawler/{cli.py, service.py, README.md}
│   │   ├── embedder/{cli.py, service.py, README.md}
│   │   ├── trainer/{cli.py, service.py, README.md}
│   │   ├── infer/{cli.py, service.py, README.md}
│   │   └── evaluate/{cli.py, service.py, README.md}
│   ├── crawl/                     # fetch, parse, robots
│   ├── embeddings/                # gemini client + cache
│   ├── models/                    # keras, calibration, persistence, registry
│   ├── pipeline/                  # common helpers, nodes (train/infer/eval), io
│   ├── storage/                   # postgres + s3 clients and repositories
│   └── utils/                     # logging, taxonomy, metrics, seed
└── tests/
```

---

## 📊 Data Contracts

### Labeled CSV
| Column               | Type        | Required | Description                                |
|----------------------|-------------|----------|--------------------------------------------|
| website              | string      | ✅        | Domain or URL                              |
| iab_labels           | list/string | ❌        | JSON list or comma‑separated IAB IDs       |
| premiumness_labels   | JSON        | ❌        | Dict {IAB_ID: 1–10}                        |
| content_text         | string      | ❌        | Optional pre‑fetched content               |

### Unlabeled CSV
| Column         | Type   | Required | Description   |
|----------------|--------|----------|---------------|
| website        | string | ✅        | Domain or URL |
| content_text   | string | ❌        | Optional      |

### Predictions JSONL (example)
```
{
"website": "cnn.com",
"geo": "US",
"categories": [
{ "id": "IAB12", "label": "News", "prob": 0.98, "score": 10 },
{ "id": "IAB14", "label": "Society", "prob": 0.76, "score": 8 },
{ "id": "IAB15", "label": "Science", "prob": 0.65, "score": 7 }
],
"generated_at": "2025-08-12T12:00:00Z"
}
```

---

## ⚙️ Installation

```
curl -sSL https://install.python-poetry.org | python3 -
poetry install
cp .env.example .env
```

### Core Env Vars
```
GEMINI_API_KEY=your_key
HTTP_USER_AGENT=Mozilla/5.0 (compatible; IABVerticalizer/1.0)
```

### Optional / Recommended
```
DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname
S3_ENDPOINT=http://localhost:9000
S3_BUCKET=verticalizer
S3_ACCESS_KEY=...
S3_SECRET_KEY=...

GEMINI_EMB_MODEL=models/text-embedding-004
GEMINI_EMB_DIM=768
GEMINI_TASK_TYPE=classification
GEMINI_EMB_DRYRUN=0
GEMINI_EMB_MAX_CALLS=0
GEMINI_EMB_RATE_LIMIT=0
```

---

## 📜 Commands

All commands share the top–level entrypoint:
```
poetry run verticalizer  [options...]
```

### 1) Crawl
Fetch & parse website content, respecting robots.txt; persist to Postgres/S3.
```
poetry run verticalizer crawl \
--in PATH_TO_CSV \
[--store-html] \
[--geo GEO_CODE] \
[--batch-size N] \
[--max-sites N]
```

### 2) Embed
Generate/reuse cached embeddings; respects rate limits and cost controls.
```
poetry run verticalizer embed \
--in PATH_TO_CSV \
[--model MODEL_NAME] \
[--store-to-s3]
```

### 3) Train
Train two-head Keras model (labels + premiumness) & calibrate.
```
poetry run verticalizer train \
--geo GEO_CODE \
--in PATH_TO_LABELED_CSV \
--version VERSION_TAG \
--out-base MODELS_DIR
```

### 4) Infer
Run inference, preparing embeddings as needed, output JSONL.
```
poetry run verticalizer infer \
--geo GEO_CODE \
--in PATH_TO_CSV \
--model PATH_TO_MODEL \
--calib PATH_TO_CALIB \
--out OUTPUT_JSONL \
[--topk N]
```

### 5) Eval
Compare predictions JSONL vs. gold JSON to generate a report.
```
poetry run verticalizer eval \
--pred PREDICTIONS_JSONL \
--gold GOLD_LABELS_JSON \
--out OUTPUT_REPORT_JSON
```

### 6) run‑pipeline (optional)
Chain all stages: crawl → embed → train → infer → eval.
```
poetry run verticalizer run-pipeline \
--geo GEO_CODE \
--in PATH_TO_LABELED_CSV \
--version VERSION_TAG \
--out-base MODELS_DIR
```

---

## 📈 Performance Tips

- Balance labeled data per geo for higher macro‑F1.
- Enable embedding cache to reduce Gemini API cost/time.
- Use rate limits and max-calls env vars for budget control.
- Prefer batch embedding for efficiency and cache locality.

---

## 🔒 Security & Compliance

- robots.txt respected by crawler.
- No PII stored.
- API keys in `.env` only.
- Optional domain allowlist for crawl compliance.

---

## 🧪 Testing & Quality

```
poetry run pytest
poetry run ruff check src --fix
```

For contributor workflow, Makefile targets, and PR process, see CONTRIBUTING.md.

---

## 📚 Specification

For scope, requirements, acceptance criteria, risks, persistence contracts, and roadmap, see spec.md.

---

## 🏷 License

Proprietary – Internal use only

---