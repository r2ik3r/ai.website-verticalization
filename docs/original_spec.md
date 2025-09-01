# Specification — IAB Website Verticalization with Geo‑specific Premiumness

## 1. Summary

**Objective:**  
Classify websites into **Interactive Advertising Bureau (IAB)** categories and generate **per‑vertical, geo‑specific Premiumness Scores** (1–10) using:

- Crawled website content
- Gemini (or other LLM/embedding) text embeddings
- A two‑head Keras model
- Probability calibration

**Key Capabilities:**
- Multi‑label IAB classification (**Tier‑1 mandatory**; Tier‑2 optional)
- Per‑vertical, geo‑aware Premiumness Scores (1–10)
- Modular, decoupled pipelines: `crawl`, `embed`, `train`, `infer`, `eval`
- Shared embedding preparation (`prepare_embeddings_for_df()`) for DRY + cache safety
- Caching + cost controls for embeddings
- Rigorous evaluation + reporting for QA
- Optional **self‑training loop** for cases with sparse labels

---

## 2. Background & Inspiration

Based in part on **Kaggle: Day‑2 Classifying Embeddings with Keras**, adapted for:
- **Multiple labels (IAB categories)**
- **Additional regression head** for Premiumness
- **Per‑geo modeling** for varying audience/market contexts
- **Production robustness**: modular CLIs, caching, logging, persistence

IAB Category Reference:  
https://docs.webshrinker.com/v3/iab-website-categories.html#iab-categories

---

## 3. Stakeholders

- **Product/Ads** — defines acceptance thresholds for classification and Premiumness, informs vertical selection.
- **Data Science / ML** — responsible for taxonomy mapping, labeling norms, training, calibration, evaluation.
- **Engineering** — owns pipeline orchestration, CLI modules, persistence layers, cost‑control mechanisms.
- **Ops / SRE** — monitors scheduled batch jobs, handles scaling, secret management, recovery.

---

## 4. Scope

**In‑Scope (Phase 1):**
- Robots‑aware crawling & content parsing
- Gemini API (or compatible LLM) embeddings
- Multi‑label classification + per‑label Premiere Score prediction
- Per‑label isotonic calibration
- Modular, independently runnable pipeline stages
- Per‑geo model training and inference

**Out‑of‑Scope for Phase 1:**
- Real‑time scoring APIs
- Tier‑2 taxonomy unless dataset size & quality justify
- UI / Human‑in‑loop tools
- 3rd‑party enrichment (traffic, SEO, brand‑safety) unless provided

---

## 5. Inputs & Outputs

### Inputs
**Labeled Geo CSV**  
- `website` — required  
- `iab_labels` — JSON array or comma‑separated string of IAB IDs  
- `premiumness_labels` — JSON `{IAB_ID: 1–10}` (geo‑specific possible)  
- `content_text` — optional pre‑fetched content_text  
- `geo` — optional if CLI flag provided

**Unlabeled Geo CSV**  
- `website` — required  
- `content_text` — optional

### Outputs
**Predictions JSONL**  
```
{
"website": "cnn.com",
"geo": "US",
"categories": [
{ "id": "IAB12", "label": "News", "prob": 0.98, "score": 10 },
{ "id": "IAB14", "label": "Society", "prob": 0.76, "score": 8 }
],
"generated_at": "2025-08-12T12:00:00Z"
}
```

**Evaluation JSON** — metrics & per‑site detail  
**Model artifacts** — `model.keras`, `calib.pkl`, config JSON

---

## 6. Functional Requirements

1. **Modular CLI stages** — `crawl`, `embed`, `train`, `infer`, `eval`; independently runnable.
2. **Crawling** — robots‑aware; incremental (by date/hash); S3/DB storage.
3. **Embedding** — cache‑first, deduplication, cost control (max‑calls, rate‑limit, dry‑run).
4. **Training** — multi‑label classification + regression head; hyperparams tunable.
5. **Inference** — load model, process CSV/DB, produce JSONL.
6. **Evaluation** — compare gold vs predictions, compute classification metrics.
7. **Premiumness Regression** — per vertical, 0–1 mapped to 1–10.
8. **Self‑Training (optional)** — pseudo‑labeling with high‐confidence threshold (≥0.85).

---

## 7. Non‑Functional Requirements

- **Accuracy target:** top‑1 macro‑F1 ≥ 0.90 (mature data)
- **Calibration:** per‑label isotonic regression
- **Performance:** batched ops; efficient DB/IO; configurable size
- **Security/Compliance:** no PII; secrets in env; robots.txt respected
- **Observability:** cache hit‑rate logs, API call counters, run durations
- **Idempotency:** repeated runs skip unchanged work

---

## 8. Data Model & Taxonomy

- **IAB Tier‑1** — required; Tier‑2 optional per sampling
- **Canonical mapping** in repo JSON (id↔label)
- Labels normalized to uppercase IAB IDs
- Premiumness clamped [1,10], normalized to [0,1] for regression head

---

## 9. Model Design

- **Input**: Gemini embeddings (`dim` via env or inferred from data)
- **Shared trunk**: Dense(512) → Dropout → Dense(…)
- **Head A (Classification)**: Dense(num_labels, sigmoid)
- **Head B (Scores)**: Dense(num_labels, sigmoid)
- **Loss**: BCE (labels) + MSE (scores)
- **Calibration**: Isotonic per‐label on validation split

---

## 10. Pipelines & Commands

All share `prepare_embeddings_for_df()` to ensure reuse.

- **crawl**: fetch+parse HTML/text → persist to DB/S3
- **embed**: load text from DB → embed via Gemini → persist vectors/meta
- **train**: prepare embeddings → build targets → fit model + calibrator → save artifacts
- **infer**: load artifacts → prepare embeddings → score & output JSONL
- **eval**: read predictions/gold → compute metrics/report

---

## 11. Evaluation & Acceptance

- **Classification:** top‑1 macro‑F1 ≥ 0.90; top‑3 ≥ 0.95 early phase
- **Calibration:** ECE/Brier improved vs uncalibrated baseline
- **Regression:** Spearman ≥ 0.6 for scores vs gold per‑vertical
- **Operational:** CLI completes without error; run logs detailed

---

## 12. Risks & Mitigation

- **Sparse labels:** use self‑training/active learning; high thresholds to avoid noise
- **Taxonomy drift:** version mapping; migration utilities
- **Robots denial:** fallback to provided content_text
- **Score meaning drift:** document per geo; preserve regressors’ calibration

---

## 13. Persistence Schema

**Tables**: sites, crawls, embeddings, models, predictions, eval_reports  
**Key contract**: `latest_text_for_site_batch()` → `{site: latest_excerpt}`  
**Object Storage**: raw HTML/text, optional vectors/artifacts

---

## 14. Operations

- **Logging**: step‐level info logs
- **Env Controls**:  
  GEMINI_EMB_DRYRUN, GEMINI_EMB_MAX_CALLS, GEMINI_EMB_RATE_LIMIT,
  GEMINI_EMB_MODEL, GEMINI_EMB_DIM, GEMINI_TASK_TYPE
- **Monitoring (Future)**: Prometheus metrics

---

## 15. Roadmap

- **Phase 1**: Tier‑1 classification + premiumness, per‑geo models
- **Phase 2**: Tier‑2 where label coverage suffices, richer metrics
- **Phase 3**: Self‑training CLI, enrichment signals, real‑time API

---

## 16. Roles/Responsibilities

- **Product:** define taxonomy scope, scores, acceptance criteria
- **DS/ML:** maintain taxonomy, labeling, training, eval
- **Engineering:** pipelines, persistence, caching, logging, cost controls
- **Ops/SRE:** schedule jobs, monitor, manage secrets

---

## 17. Glossary

- **IAB:** Interactive Advertising Bureau taxonomy
- **Premiumness Score:** 1–10 per‑vertical metric of ad inventory quality
- **Calibration:** aligning predicted probabilities with empirical truth
- **Self‑Training:** pseudo‐labeling + retraining cycle

---

## 18. Cross‑References

- **How to run**: see [README.md](./README.md)
- **Contributor workflow**: see [CONTRIBUTING.md](./CONTRIBUTING.md)
- **Module details**: see `apps/*/README.md`

---

## 19. Acceptance Checklist

- All CLIs runnable independently
- train/infer/eval use shared embedding helper
- latest_text_for_site_batch correct mapping
- Cache-first embedding; env cost controls functional
- Accuracy, calibration meet target or baseline documented
- Docs in README/module READMEs match code

---

***



[1] https://www.kaggle.com/code/markishere/day-2-classifying-embeddings-with-keras


```aiexclude
Original Prompt

Please refer https://www.kaggle.com/code/markishere/day-2-classifying-embeddings-with-keras throughly.

I am creating this project with such similar requirement. Please read them carefully

My requirements:

I need to verticalize websites. Verticals are standatd IAB categories defined at https://docs.webshrinker.com/v3/iab-website-categories.html#iab-categories

Example:
Site Vertical1 Vertical2 Vertical3 Vertical1 IAB Vertical2 IAB Vertical3 IAB Premiumness Score
weather.com News Science Travel IAB12 IAB15 IAB20 10
ebay.com Shopping Hobbies&Interests Technology&Computing IAB22 IAB9 IAB19 10
cnn.com News Law,Government,&Politics Society IAB12 IAB11 IAB14 10
realtor.com RealEstate Travel Business IAB21 IAB20 IAB3 10

So, weather.com is categorized as verticals in the order of News, Science, and Travel. It is also given a Premiumness Score of 10.

Here Premiumness Score is the same for each vertical while it should be different for every verticals. And also, it will be geo based i.e. websites may have different scores for verticals in different geographies.

Input is a list of websites.
Output should be a Map<Website, Map<IAB Vertical, Score>>

I want the model to give accurate result and the accuracy should be more than 90%.

1. Use modern libraries in Python (Poetry, langgraph)
2. Use Scikit learn, Keras, Panda, Numpy, etc powerful libraries to achieve this result.
3. Use classification model to classify a wbsite into verticals and give Premiumness Score.
4. Score shuld be in range 1 to 10.
5. Use powerful LLM (either opensource or gemini; I have gemini API) for model training and classification.
6. Generate all required packages and files. They should be correct, production ready, efficient, and accurate.
7. I can provide input files for each geo.
8. Take input file and generate output files in such a way that I can feed outfile as input to the code for training, validation, and classification.
9. Create a README.md file with all information
10. This is a company project so not an open source project.
11. Add if I may have missed something but you find very usefull for this project.



How can I (and you) make sure that my following requirements are met?
1. I want to productionize it but by making it modular i.e. pipelines are decoupled and can be run separately as required.
2. My thoughts are that, crawling is not always required, so we can persist the crawled websites. We can create a module to crawl websites on requirement basis and on batch basis i.e. both incremental websites and batch websites. Store the crawled output in a Database. You pick the right database for this.
3. Training (including embedding, etc) will be a separate module. i.e. if we feel that we need to train the model again then we can do it easily. We can upgrade models and then train again. We can tune models and then train again. We can increase and tune out training dataset and train the model again. So training will become a separate module.
4. Then prediction (or infer) become a separate module i.e. we can predict verticals and premiumness score for new websites when needed or websites in batch when needed.
5. Evaluation (or compare predictions) become a new module i.e. we can compare predictions for a new website or websites in batch.
6. Also, model should train on the labeled verticals for a website and predict verticals for a website.
7. Then, model should train on the labeled premuimness score (premium publisher score) for a website and should predict premuimness score for a website wrt a vertical. Please note that this project is being done for ad-tech domain where premuimness score defines if a website has premium ad inventory or not (i.e. genuine or organic users visits the websites and brands (or advertisers) know the website as a safe and premium website). So if you can collect how many unique users, organic users, premuimness based of SEO, etc then it will be helpful to define premuimness score (or website score).
8. Update README.md for each module.
9. By separate module, I mean one project but I can run separate part of the complete pipeline as needed. And each module should be correct, complete, efficient, cached, optimized, cost-efficient, logged, monitored, input and output friendly.
10. I need architecture diagram of the the complete pipeline.
```