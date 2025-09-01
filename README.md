# Website Verticalizer

Production-grade system to classify websites into IAB categories with calibrated probabilities, optional Premiumness scores, and per-geo/version artifact management. The pipeline supports versioned IAB taxonomies, hierarchy-consistent outputs, multi-URL site aggregation, and optional ensembling across multiple model paths.

## Overview

Website Verticalizer ingests labeled content to train a multilabel classifier over website/page text embeddings, then runs inference to emit top‑k IAB predictions per site. It includes:
- Versioned IAB taxonomy loader (v3.0 and v2.2), normalization by ID/label, and optional cross‑version mapping.
- Hierarchy consistency at inference by enforcing parent floors, plus optional parent augmentation in top‑k.
- Embedding clients with cache‑first behavior (Gemini implemented) and an optional sentence‑transformers pathway for ensembling.
- Per‑label isotonic calibration for reliable probabilities and stable top‑k ranking; optional focal loss and early stopping for long‑tail labels.
- Multi‑URL aggregation to turn page‑level predictions into site‑level decisions.

## Data contracts

All CSVs are UTF‑8. Minimal schemas:
- Labeled training CSV
  - website: domain like cnn.com.
  - iablabels: JSON array of uppercase IAB IDs (e.g., ["IAB12","IAB1"]); comma‑separated also accepted at training time.
  - Optional: contenttext free text if already gathered; otherwise, crawler will fetch latest excerpt.
  - Optional: premiumnesslabels JSON object mapping IAB ID to 1–10 integer (kept orthogonal to probabilities).
- Unlabeled inference CSV (single URL per site)
  - website: domain.
  - Optional: contenttext; if missing, crawler + embedder reuse latest excerpt.
- Unlabeled inference CSV (multi‑URL per site)
  - website: domain.
  - url: fully qualified page URL; multiple rows per website for aggregation.
- Predictions JSONL (output)
  - One JSON per line: { website, categories: [{ id, label, prob }] }; top‑k size configurable; parents may be appended for hierarchy readability.

Taxonomy assets required:
- Place JSONs under data/taxonomy/{v3,v2_2}/:
  - id_to_label.json, label_to_id.json, graph.json (parent -> [children])
- Optional cross‑version maps: data/taxonomy/map/{v2_2_to_v3.json, v3_to_v2_2.json}

## Installation

- Python 3.10+. Install via Poetry and configure environment variables.
- Key env variables:
  - Embeddings: GEMINI_API_KEY, GEMINI_EMB_MODEL, GEMINI_EMB_DIM, GEMINI_EMB_RATE_LIMIT, GEMINI_EMB_MAX_CALLS, GEMINI_EMB_DRYRUN; optional SENTENCE_TFM_MODEL/SENTENCE_TFM_DIM.
  - Storage: DB_DSN (Postgres), S3_ENDPOINT/S3_BUCKET/S3_ACCESS_KEY/S3_SECRET_KEY/S3_REGION (optional).
  - Crawler: HTTP_USER_AGENT, HTTP_TIMEOUT.

Setup
- poetry install
- cp .env.dev.template .env and fill required keys. # cost-efficient, real embeddings
- cp .env.prod.template .env and fill required keys. # accuracy-first production

## How to run (Make targets)

Train once, then predict on demand. These targets wrap the canonical CLI commands. Paths and flags can be adjusted at the top of the Makefile (GEO, VERSION, IAB_VERSION, data/model directories).

- Ingest Kaggle IAB dataset to labeled CSV
  - make run-ingest
- Train and save artifacts
  - make run-train
- (Optional) Crawl multiple URLs per site for richer inference
  - make run-crawl
- Predict (single model)
  - make run-infer
- Predict (ensemble with site‑level aggregation)
  - make run-infer-ensemble
- End‑to‑end flow
  - make run-all

Outputs
- Models and calibrators under models/<GEO>/<VERSION> (model.keras, calib.pkl).
- Predictions at out/preds.jsonl with top‑k categories per website.

## Input data examples

- Labeled CSV
  - website: cnn.com
  - iablabels: ["IAB12","IAB1"]
  - contenttext: “Breaking news coverage …”
- Inference CSV (multi‑URL)
  - website: cnn.com
  - url: https://www.cnn.com/politics/article-1.html

## Operational guidance

- Embeddings: cache‑first; maximize cache hits; use DRYRUN/MAX_CALLS for cost control; disable DRYRUN for production vectors.
- Crawling: robots.txt‑aware with courtesy delay and configurable UA/timeout; failures still record crawl rows.
- Calibration: per‑label isotonic when positives ≥ 5; applied only to classification probabilities.
- Multi‑URL aggregation: mean or softmax_mean before top‑k; recommend 3–10 URLs per site.
- Hierarchy consistency: enforce parent floors and optionally append parents to top‑k.

## Repository layout

- src/verticalizer/
  - apps/{crawler,embedder,trainer,infer,evaluate}: CLIs and services.
  - embeddings/: Gemini and optional sentence‑transformers clients; persistent cache.
  - models/: Keras heads, calibration, persistence/registry.
  - pipeline/: training/inference nodes, ensemble/postprocess utilities, IO helpers.
  - utils/: taxonomy loaders (versioned), metrics, logging, seed.
  - storage/: Postgres and S3/MinIO repositories/clients.
- data/taxonomy/: IAB assets as described above.

## License

Proprietary — internal use only.

