# Evaluate (apps/evaluate)

Compares predictions JSONL to gold labels and writes a summary report for QA and regression checks.

See also
- Architecture, environment, and commands: README.md
- Specification (requirements, acceptance criteria): spec.md

---

## What it does

- Reads predictions (JSONL) and gold labels (JSON).
- Computes overlap-oriented summary (e.g., match rate).
- Emits a JSON report with summary and per-site details.

---

## CLI

```
poetry run verticalizer eval \
--pred PREDICTIONS_JSONL \
--gold GOLD_LABELS_JSON \
--out OUTPUT_REPORT_JSON
```

Inputs
- Predictions JSONL (from infer).
- Gold JSON mapping `{website: {IAB_ID: score_or_1}}`.

Outputs
- Eval report JSON with:
  - `summary`: sites, matched_any, match_rate
  - `details`: per-site gold/pred overlaps

---

## Extending metrics

- For full multilabel metrics (precision/recall/F1, top‑k), you can call `pipeline/nodes.py::evaluate()` using a labeled CSV and model artifacts. Consider exposing this as a secondary eval command in the future.

---

## Environment

- None beyond file I/O, unless you extend to DB logging.

---