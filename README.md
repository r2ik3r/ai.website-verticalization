
# 📊 IAB Verticalizer  
**Production-grade classification system for website content into IAB categories with geo-specific Premiumness Scores**

---

## 📌 Overview

**IAB Verticalizer** is a production-ready ML system for:

- Classifying websites into **IAB categories** (Tier-1 & optionally Tier-2).
- Assigning **geo-specific Premiumness Scores** (1–10) per category.
- Operating on **labeled datasets**, unlabeled datasets (via crawling, embedding, inference), or a hybrid **self-training loop**.
- Combining **crawled website content** + **Gemini embeddings** + **Keras classifier with calibration**.
- Achieving **production-level robustness, modularity, and reproducibility**.

Inspired by **Kaggle Day 2 — Classifying embeddings with Keras**, extended for multi-label, multi-geo, production deployments.

---

## 🚀 Features

✅ **Multi-label Keras classification** using semantic embeddings  
✅ **Gemini API integration** for embeddings  
✅ **Website crawling** with `robots.txt` compliance  
✅ **Geo-specific Premiumness Scores** (1–10)  
✅ **Probability calibration** via isotonic regression  
✅ **Self-training bootstrap** for sparse labels  
✅ **Configurable LangGraph pipelines**  
✅ **File-driven I/O** for chaining outputs to retraining  
✅ **Scalable for production** with logging, retries & caching

---

## 🏗 Architecture & Workflow

```
           ┌────────────┐
           │ Input Data │
           │ (Labeled/  │
           │ Unlabeled) │
           └─────┬──────┘
                 │
        ┌────────▼─────────┐
        │  Content Fetcher │
        │  (Crawl + Parse) │
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Gemini Embedder  │
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Keras Classifier │
        │  + Score Head    │
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Calibration      │
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
┌────▶│ Predictions JSON │
│     └──────────────────┘
│
│ Self-Training Loop (if labels sparse)
│  ┌─────────────────────────────┐
└──│ Crawl unlabeled → Predict → │
│ High-confidence labels →    │
│ Augment training → Retrain  │
└─────────────────────────────┘
```

**Flow:**
1. **Data** – Either labeled CSV (with or without content) or just domains.
2. **Crawler/Parser** – Robots-aware fetch of HTML, convert to clean text.
3. **Embedding** – Gemini API creates semantic vectors.
4. **Classification Model** – Keras MLP outputs multi-label category probabilities + score bins.
5. **Calibration** – Isotonic regression refines probabilities.
6. **Output** – JSONL with top categories & Premiumness scores.
7. **Self-Training Loop** – Bootstraps training on high-confidence unlabeled predictions.

---

## 📂 Project Structure

```
verticalizer/
├── pyproject.toml
├── README.md
├── .env.example
├── src/verticalizer/
│   ├── cli.py
│   ├── crawl/            # fetch, parse, robots
│   ├── embeddings/       # gemini client + cache
│   ├── features/         # crawl+embed orchestrator
│   ├── models/           # keras, calibration, persistence
│   ├── pipeline/         # train, infer, self-train
│   ├── utils/            # logging, taxonomy, metrics, seed
│   └── data/taxonomy/    # IAB mappings
└── tests/
```

---

## 📊 Data Contracts

**Labeled CSV**

| Column               | Type        | Required | Description                                |
|----------------------|-------------|----------|--------------------------------------------|
| `website`            | string      | ✅        | Domain or URL                              |
| `iab_labels`         | list/string | ❌        | JSON list or comma-separated IAB IDs/names |
| `content_text`       | string      | ❌        | Optional                                   |
| `premiumness_labels` | JSON        | ❌        | Dict {IAB_ID: 1–10}                        |

**Unlabeled CSV**

| Column         | Type   | Required | Description   |
|----------------|--------|----------|---------------|
| `website`      | string | ✅        | Domain or URL |
| `content_text` | string | ❌        | Optional      | 

**Prediction JSONL Example**
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
cp .env.example .env  # set GEMINI_API_KEY
```

**Env Vars**
```
GEMINI_API_KEY=your_key
HTTP_USER_AGENT=Mozilla/5.0 (compatible; IABVerticalizer/1.0)
```

---

## 📜 Commands

**Train**
```
poetry run verticalizer train \
  --geo US \
  --in data/us_labeled.csv \
  --model-out models/us_model.keras \
  --calib-out models/us_calib.pkl \
  --report reports/us_val.json
  
poetry run verticalizer train \
  --geo IN \
  --in data/in_labeled.csv \
  --model-out models/in_model.keras \
  --calib-out models/in_calib.pkl \
  --report reports/in_val.json
```

**Validate**
```
poetry run verticalizer validate \
--geo US --in data/us_labeled.csv \
--model models/us_model.keras \
--calib models/us_calib.pkl \
--report reports/us_validation.json
```

**Infer**
```
poetry run verticalizer infer \
  --geo US \
  --in data/us_labeled.csv \
  --model models/us_model.keras \
  --calib models/us_calib.pkl \
  --out outputs/us_predictions.jsonl \
  --topk 26
  
poetry run verticalizer infer \
  --geo IN \
  --in data/in_labeled.csv \
  --model models/in_model.keras \
  --calib models/in_calib.pkl \
  --out outputs/in_predictions.jsonl \
  --topk 26
```

**Self-Train**
```
poetry run verticalizer self-train \
--geo US --seed data/small_seed.csv \
--unlabeled data/unlabeled.csv \
--iterations 3 \
--model-out models/us_model.keras \
--calib-out models/us_calib.pkl \
--report reports/us_selftrain.json
```

---

## 📈 Performance Tips

- Aim for **balanced labeled data per geo** for 90%+ macro-F1.
- Use **Tier-2** only with large, clean datasets.
- Enable **embedding cache** to reduce Gemini costs/time.
- Integrate **active learning**: review low-confidence predictions before retraining.

---

## 🔒 Security & Compliance

- `robots.txt` respected by crawler.
- No PII or sensitive content stored.
- API keys in `.env` only.
- Optional domain allowlist for compliance.

---

## 🧪 Testing
```
poetry run pytest tests/
```

---

## ❓ FAQ
**Q:** No labeled data?  
**A:** Bootstrap with a small seed + self-training.

**Q:** Different scores for each vertical?  
**A:** Yes, via calibrated scoring head.

---

## 📚 References
- Kaggle: *Day 2 - Classifying embeddings with Keras*
- Webshrinker: *IAB Website Categories*
- LangGraph: *ML Orchestration*

---

## 🏷 License
**Proprietary – Internal use only**
