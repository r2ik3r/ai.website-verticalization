# Crawler (apps/crawler)

Robots-aware website fetcher that extracts text content for downstream embedding and modeling. Persists crawl metadata to Postgres and optionally stores raw HTML/text to object storage.

See also
- Architecture, environment, and commands: README.md
- Specification (requirements, acceptance criteria): spec.md

---

## What it does

- Reads a CSV of sites (column: `website`).
- Respects robots.txt and configurable user-agent/timeouts.
- Fetches HTML, parses to text, computes content hash.
- Persists:
  - Postgres: `sites`, `crawls` rows with timestamps and text_excerpt.
  - S3/MinIO (optional): raw HTML/text blobs.
- Idempotent and incremental via `last_crawled_at` and `content_hash`.

---

## CLI

```
poetry run verticalizer crawl \
--in PATH_TO_CSV \
[--store-html] \
[--geo GEO_CODE] \
[--batch-size N] \
[--max-sites N]
```

Inputs
- CSV with at least `website` column.
- Optional filters like `--geo` if your CSV includes such columns.

Outputs
- Postgres rows in `sites` and `crawls`.
- Optional S3 objects per site/page if `--store-html` is provided.

---

## Environment

- DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db
- S3_ENDPOINT=http://localhost:9000
- S3_BUCKET=verticalizer
- S3_ACCESS_KEY=...
- S3_SECRET_KEY=...
- HTTP_USER_AGENT=Mozilla/5.0 (compatible; IABVerticalizer/1.0)

---

## Operational Notes

- Clear logs for robots-allowed/denied, success/fail counts.
- Safe to re-run — unchanged content should be skipped.
- Ensure DB migrations/tables are created (storage layer handles this).

---