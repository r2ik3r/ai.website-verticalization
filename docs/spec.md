Project Specification: Website Verticalization with IAB Categories and Geo-specific Premiumness

1. Summary
- Objective: Classify websites into IAB categories and produce per-vertical, geo-specific Premiumness Scores from 1–10, using crawled content, embeddings, ML classification, and calibration.
- Key Capabilities:
  - Multi-label IAB classification (Tier-1; extensible to Tier-2).
  - Per-vertical, geo-aware Premiumness Score (1–10).
  - Self-training pipeline that bootstraps when labeled data is limited.
  - File-driven inputs/outputs for training, validation, inference, and evaluation at scale.

2. Stakeholders
- Product/Ads: Needs accurate vertical categorization and premium inventory scoring for targeting and allowlists.
- Data Science/ML: Owns taxonomy mapping, training, calibration, and evaluation.
- Engineering: Owns pipelines, deployment, and batch operations.

3. Scope
- In-scope:
  - Domain crawling (robots-aware), content parsing, embeddings via Gemini.
  - Keras-based multi-label classifier with per-vertical scoring head.
  - Probability calibration (isotonic) for reliable ranking.
  - Self-training loop with high-confidence pseudo-labels.
  - Per-geo training and inference.
- Out-of-scope (Phase 1):
  - Real-time streaming classification.
  - Tier-2 taxonomy without sufficient labeled data.
  - Human-in-the-loop UI (optional later).
  - Third-party enrichment (e.g., SimilarWeb) unless provided.

4. Inputs and Outputs
- Inputs:
  - Per-geo labeled CSV: website, iab_labels(JSON array or comma-separated), premiumness_labels(JSON map {IAB_ID:1–10}), geo, optional content_text.
  - Unlabeled CSV: website, optional content_text.
- Outputs:
  - Predictions JSONL per geo: {website, geo, categories: [{id, label, prob, score(1–10)}], generated_at}.
  - Validation reports (JSON): top1/top3 accuracy, macro/micro F1, calibration metrics.
  - Augmented training CSVs from self-training.

5. Functional Requirements
- Classifier must predict probabilities for all IAB Tier-1 categories (26 classes) and return a ranked list (configurable topK; default 26).
- Scoring head must output a score per vertical, mapped to 1–10.
- Scores and probabilities must be geo-specific where different per-geo models are trained.
- Self-training must:
  - Crawl and embed unlabeled sites.
  - Create pseudo-labels for categories with probability≥threshold (configurable; default 0.85).
  - Retrain with augmented data while preventing collapse (e.g., by mixing seed set each iteration).
- Pipeline must honor robots.txt and support configurable user-agent and timeouts.
- Embeddings must be cached for reproducibility and cost efficiency.

6. Non-Functional Requirements
- Accuracy:
  - Target top-1 macro-F1≥0.90 given high-quality labeled data; otherwise aim top-3≥0.95 in early iterations.
- Reliability:
  - Probability calibration (per-label isotonic) on validation split.
  - Deterministic seeds for reproducible runs.
- Performance:
  - Batched crawling and embedding with caching.
  - Train batch size/epochs configurable.
- Security/Compliance:
  - No PII storage; robots-aware crawling; API keys in environment.
- Operability:
  - CLI with commands for train/validate/infer/self-train.
  - Clear logs and error handling.

7. Data Model and Taxonomy
- IAB Taxonomy: Tier-1 (IAB1–IAB26) required; Tier-2 optional with adequate labels.
- Canonical ID↔label mapping stored in repo.
- Normalization layer maps incoming labels to canonical IAB IDs.

8. Model Design
- Inputs: Gemini text embeddings from crawled or provided content_text.
- Architecture:
  - Shared MLP trunk over embeddings.
  - Head A (labels): Dense(num_labels, sigmoid) for multi-label probabilities.
  - Head B (scores): Dense(num_labels, sigmoid) for per-vertical scores (normalized 0–1 → scaled to 1–10).
- Loss:
  - Binary cross-entropy for labels.
  - MSE (or Huber) for scores.
- Calibration:
  - Per-label isotonic regression on validation logits for labels head.

9. Pipelines and Commands
- Train:
  - poetry run verticalizer train --geo  --in data/_labeled.csv --model-out models/_model.keras --calib-out models/_calib.pkl --report reports/_val.json
- Infer (all verticals ranked):
  - poetry run verticalizer infer --geo  --in data/_unlabeled.csv --model models/_model.keras --calib models/_calib.pkl --out outputs/_predictions.jsonl --topk 26
- Validate:
  - poetry run verticalizer validate --geo  --in data/_labeled.csv --model models/_model.keras --calib models/_calib.pkl --report reports/_eval.json
- Self-train:
  - poetry run verticalizer self-train --geo  --seed data/_seed.csv --unlabeled data/_unlabeled.csv --iterations 3 --model-out models/_model.keras --calib-out models/_calib.pkl --report reports/_selftrain.json

10. Evaluation and Acceptance Criteria
- Correctness:
  - Classification: top-1 macro-F1≥0.90 on held-out validation for mature geos/datasets; otherwise document baseline and improvement targets.
  - Calibration: ECE or Brier improvement vs. uncalibrated.
  - Scoring: Spearman correlation≥0.6 between predicted and provided premiumness scores per vertical on validation where available.
- Functional:
  - For an input site, inference returns ordered list of all verticals with prob and score(1–10).
  - For geos with seed labels, self-training produces improved validation metrics across iterations or early-stops safely.
- Operational:
  - CLI commands execute end-to-end with clear logs and non-zero exit on failure.
  - Embedding cache hit rate reported.

11. Risks and Mitigations
- Sparse labels per vertical/geo:
  - Use self-training + active learning; enforce high-confidence thresholds.
- Taxonomy drift:
  - Lock taxonomy files; version maps; add migration utilities.
- Crawl failures/robots denial:
  - Fallback to content_text; track coverage and allow manual injection.
- Score interpretation:
  - Ensure per-vertical scores are calibrated to 1–10; document meaning by geo.

12. Deliverables
- Codebase (src/verticalizer/*) with CLI.
- README.md for setup and operations.
- docs/spec.md (this document).
- Example data: data/_labeled.csv, _unlabeled.csv.
- Reports: validation and self-training metrics.
- Models and calibrators per geo.

13. Open Questions
- Tier-2 rollout criteria (minimum samples/class?).
- Business definition of “premiumness” per vertical when no labels exist (proxy acceptance?).
- SLA/throughput requirements for batch classification.

Do you need a prompt?
- If “prompt” means an LLM system prompt for agents: not required for the core pipeline unless adding an LLM QA node or auto-labeling beyond embeddings/classifier; the current system runs fully with the spec and README.
- If “prompt” means a project brief to kick off internal work: use the above spec as your single-source write-up. Keep README for setup and ops, and this spec for scope and acceptance.

If you want, I can also generate:
- A one-page executive brief (non-technical).
- A JIRA-ready epic with stories/tasks derived from this spec.
- A short architecture diagram image and embed-ready text snippet for your docs.