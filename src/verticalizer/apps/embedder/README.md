# Embedder (apps/embedder)

Generates semantic embeddings for crawled content using Gemini, with cache-first behavior and cost controls. Embeddings are reused across training and inference.

See also
- Architecture, environment, and commands: README.md
- Specification (requirements, acceptance criteria): spec.md

---

## What it does

- Reads a CSV of sites (`website`).
- Loads latest text per site from Postgres (and S3 if needed).
- Embeds text via Gemini with:
  - Cache-first lookup (avoid re-embedding unchanged text)
  - Deduplication across identical texts
  - Dry-run and rate-limit/max-calls controls
- Persists embedding metadata (and optional vectors) to storage.

---

## CLI

```
poetry run verticalizer embed \
--in PATH_TO_CSV \
[--model MODEL_NAME] \
[--store-to-s3]
```

Inputs
- CSV with `website` column.
- Assumes crawler populated text previously (or `content_text` provided).

Outputs
- Embedding metadata rows in Postgres (`embeddings`).
- Optional vector blobs to S3 if enabled.

---

## Environment

- GEMINI_API_KEY=...
- GEMINI_EMB_MODEL=models/text-embedding-004
- GEMINI_EMB_DIM=768
- GEMINI_TASK_TYPE=classification
- GEMINI_EMB_DRYRUN=0      # 1 to skip API calls, return zeros
- GEMINI_EMB_MAX_CALLS=0   # 0=unlimited
- GEMINI_EMB_RATE_LIMIT=0  # QPS
- DATABASE_URL=postgresql+psycopg2://...

---

## Cost and Reliability

- Cache-hit rate is logged.
- MAX_CALLS and RATE_LIMIT prevent runaway spend.
- Deduplication avoids repeated embedding of identical text.

---