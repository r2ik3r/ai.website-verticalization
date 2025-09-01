# Website Verticalizer — Overview & Quickstart

## Summary
The project classifies websites into IAB Tier‑1 categories and generates per‑vertical, per‑geo Premiumness Scores (1–10) by combining robots‑aware crawling, semantic embeddings, a two‑head Keras network, and per‑label isotonic calibration, delivered as modular batch pipelines with cache‑first cost controls and durable persistence.
It generalizes a Keras‑on‑embeddings pattern to multilabel classification plus a score head, adds per‑geo versioning and idempotent artifact management, and guarantees deterministic runs suitable for production rollouts and rollbacks.

## Stakeholders
- Product/Ads: sets acceptance thresholds, governs vertical scope, defines score semantics and per‑geo release policies, and owns rollout criteria and deprecation rules.
- DS/ML: manages taxonomy governance, labeling standards, training/calibration/evaluation protocols, and ongoing drift/correlation monitoring across geos and versions.
- Eng/Ops: owns orchestration, CLIs, storage contracts, provider abstraction, batch cost controls, observability, rollback tooling, and SRE runbooks for continuity.

## Scope
- In: robots‑aware crawling, semantic embeddings, multilabel IAB Tier‑1 classification, per‑label probability calibration, joint per‑vertical geo‑specific scores, modular CLIs, per‑geo versioning and registry.
- Out (Phase 1): real‑time APIs, Tier‑2 expansion unless label coverage is sufficient, human‑in‑the‑loop UI, and third‑party enrichments unless explicitly provided by Product.

## Data contracts
- Labeled CSV: website (required), iablabels (JSON array or comma‑separated IAB IDs), premiumnesslabels (JSON map IABID→1..10), contenttext (optional), geo (optional); IDs normalize to uppercase IABxx and invalid labels are logged and rejected.
- Unlabeled CSV: website (required) with optional contenttext; used for inference or classification‑only training to leverage mixed supervision when scores are sparse.
- Predictions JSONL: one row per website with geo, categories[{id,label,prob,score}], generatedat ISO‑8601; inference supports path‑based or registry‑based model selection for operational flexibility.

## Pipelines
A shared helper, prepareembeddingsfordf, enforces crawl/embed reuse, assembles float32 X, and underpins stage‑independent, idempotent CLIs; blobs are stored in object storage and metadata/artifacts in Postgres for traceability.
CLI surface: crawl, embed, train, infer, eval, plus run‑pipeline to convert Excel→CSV/JSON, train, infer, and compare in one composed flow for rapid QA and acceptance testing.

## Architecture
The canonical flow is Inputs (CSV/Excel) → Crawl (robots‑aware) → Embed (cache, dedup, batch) → Train (two‑head Keras, per‑label isotonic) → Artifacts (model.keras, calib.pkl) → Infer (predict probs+scores; apply calibration) → Output JSONL.
Two‑head details live in the head‑specific deep dives; calibration applies only to classification probabilities and not to the integerized Premiumness output, preserving monotone mapping and operational simplicity.

## Persistence
- Tables: sites, crawls, embeddings, models, predictions, evalreports; object store holds raw HTML, embedding bytes, and model/calibrator artifacts; batch text contract latesttextforsitebatch(site) returns the latest excerpt.
- Governance: per‑geo versions are immutable once released; rollbacks select prior registry entries; artifact paths, checksums, and config JSON are recorded for deterministic provenance and audits.

## Evaluation gates
- Classification: require $$F1_{\text{macro}}$$ top‑1 ≥ 0.90 and top‑3 ≥ 0.95 on mature data; calibrated ECE/Brier must improve vs uncalibrated baselines; per‑label thresholds or top‑k policy is documented and frozen for acceptance.
- Scores: require per‑vertical Spearman ≥ 0.6 where label counts suffice; report sample counts per label and suppress correlation metrics where below threshold; monitor per‑geo distribution drift across versions.

## Risks and handling
- Sparse labels: enable high‑confidence self‑training on classification; mask missing scores in the loss; maintain conservative thresholds to avoid propagating noise; version taxonomy and provide migration utilities.
- Robots denial or fetch failures: fall back to provided contenttext, record crawl status, and default to zero‑vector embedding with warnings; guard budgets with DRYRUN/MAXCALLS to keep pipelines healthy.

## Data quality (no duplicates, no staleness, no clutter)
- Deduplication: canonicalize site identifiers, track latest contenthash, and cache embeddings keyed by normalized text+model; dedup identical strings prior to embedding; optionally enable near‑dup scan (e.g., simhash/MinHash) for very large batches.
- Freshness: choose latest excerpt per site by fetchedat; run incremental crawl keyed by contenthash; alert when embeddings are stale relative to lasthash; reconcilers verify “latest text” contracts.
- De‑cluttering: readability removal of scripts/styles and metadata distillation; predictable excerpt truncation; log extremely short/empty texts to help fix upstream content issues.

## Embeddings — providers and cost
- Default: Gemini text‑embedding‑004 (dimension via env), operated cache‑first with DRYRUN, MAXCALLS, QPS limiter, retries for 429/503, and zero‑vector fallback with hashed snippets for traceability.
- Provider abstraction: config permits alternate providers or local models if specify dimension/normalization; batching/dedup reduce per‑call overhead and cost without changing downstream contracts.

## Thresholding and calibration policy
- Thresholding: freeze one release policy—global 0.5 threshold, per‑label thresholds from validation ROC, or top‑k only; default is top‑k for ranking, with calibrated probabilities exposed for QA.
- Calibration: per‑label isotonic fitted on validation; applied only to classification probabilities at inference; Premiumness is discretized $$1..10$$ without separate calibration, with a documented option to add score calibration post‑label growth.

## Ops, stability, and reliability
- Crawler: robots.txt‑aware with courtesy delay and configurable UA/timeout; optional raw HTML storage behind a flag plus retention/TTL; no PII collected intentionally; secrets via env/secret manager.
- Observability: structured logs for cache hit‑rate, API calls, QPS limiting, and durations; alerts for quota exhaustion, low cache hit‑rate, high zero‑vector share, and stale embeddings vs lasthash.
- SRE runbooks: quota exceeded → switch to batch/offline and backoff; DB migration failures → DDL replay and canary; object‑store consistency → re‑upload/verify by checksum with registry reconciliation.

## Roadmap
- Phase 1: Tier‑1 classification with joint scores, per‑geo models, calibrated probabilities, cost controls, and batch embedding; documentation and contracts stable.
- Phase 2: Tier‑2 categories where labels suffice; optional score calibration/quantile mapping; more embedding providers; richer eval slices and bias/coverage audits.
- Phase 3: self‑training CLI, optional third‑party signals (traffic/SEO/brand‑safety) if provided, and a real‑time scoring API after latency and cost targets are validated.

## Acceptance checklist (frozen)
- Embeddings: provider abstraction with Gemini plus at least one alternate path; batch/dedup semantics; env‑driven model/dimension; cache hit‑rate/QPS caps logged; zero‑vector rate within budget.
- Classification: macro‑F1 and top‑k reported; ECE/Brier improvements demonstrated; threshold/top‑k policy pinned; seeds fixed for reproducibility and consistent acceptance comparisons.
- Scores: per‑vertical Spearman and distribution stats emitted; masked loss on missing scores; $$1..10$$ mapping is deterministic and documented for downstream ranking.
- Persistence: tables auto‑created if missing; artifacts saved with recorded paths and (recommended) checksums; predictions/eval inserted transactionally; object‑store layout follows spec.
- Ops: robots respected; allow/deny lists maintained; recovery playbooks validated; minimal metrics and alerts in place; clear retention policies for HTML and vectors.

Quickstart CTAs: Run Train -  Run Infer -  Open Predictions -  Compare vs Gold -  View Env Keys -  Open Architecture Diagram
