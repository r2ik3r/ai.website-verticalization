# Trainer (apps/trainer)

Trains the two-head Keras model:
- Head 1: multilabel IAB classification (sigmoid per class).
- Head 2: per-vertical Premiumness regression (normalized 0–1, reported as 1–10).

Uses shared embedding preparation and probability calibration.

See also
- Architecture, environment, and commands: README.md
- Specification (requirements, acceptance criteria): spec.md

---

## What it does

- Reads labeled CSV (`website`, `iab_labels`, `premiumness_labels`).
- Prepares embeddings via `pipeline/common.py::prepare_embeddings_for_df()`.
- Builds targets:
  - `iab_labels`: multi-hot vector (accepts JSON list or comma-separated string of IAB IDs).
  - `premiumness_labels`: dict of `{IAB_ID: 1–10}`, normalized to 0–1 for training.
- Trains model and fits calibrator if enough positives.
- Saves artifacts under a model registry path.

---

## CLI

```
poetry run verticalizer train \
--geo GEO_CODE \
--in PATH_TO_LABELED_CSV \
--version VERSION_TAG \
--out-base MODELS_DIR
```

Inputs
- Labeled CSV with columns:
  - `website` (required)
  - `iab_labels` (list or string; JSON list or comma-separated)
  - `premiumness_labels` (JSON dict of {IAB_ID: 1–10})
  - `content_text` (optional; crawler will supply text otherwise)

Outputs
- Model: `MODELS_DIR/{geo}/{version}/model.keras`
- Calibrator: `MODELS_DIR/{geo}/{version}/calib.pkl`
- Optional: config/metrics JSON

---

## Environment

- DATABASE_URL, GEMINI_* variables for embedding prep.
- Training hyperparameters can be added as flags or envs if exposed.

---

## Notes

- Embedding step is cache-first; reruns should be cheap.
- Ensure taxonomy (IAB IDs/labels) is consistent across datasets and inference.
- Version tags like `v1`, `2025-08` are recommended for reproducibility.

---