# Infer (apps/infer)

Generates predictions for websites: top‑K IAB categories and a Premiumness score per category.

See also
- Architecture, environment, and commands: README.md
- Specification (requirements, acceptance criteria): spec.md

---

## What it does

- Loads a specific model version and calibrator.
- Prepares/reuses embeddings via shared helper.
- Outputs JSONL with categories, probabilities, and 1–10 scores.

---

## CLI

```
poetry run verticalizer infer \
--geo GEO_CODE \
--in PATH_TO_CSV \
--model PATH_TO_MODEL \
--calib PATH_TO_CALIB \
--out OUTPUT_JSONL \
[--topk N]
```

Inputs
- CSV with `website` column.
- Model and calibrator files from trainer artifacts.

Outputs
- JSONL, one object per site:
  - `website`, `categories`[{`id`,`label`,`prob`,`score`}], `generated_at`

---

## Environment

- DATABASE_URL, GEMINI_* for embedding preparation.
- DRYRUN can be used to avoid new API calls when auditing.

---

## Notes

- For strictly offline reads, consider a read-only flag to skip crawl/embed and fail if embeddings are missing.
- Typical top‑K depends on taxonomy breadth; 3–26 are common values.

---