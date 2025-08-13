# Inference Module

Purpose
- Predict verticals and premiumness for websites.

Input
- CSV (website + content_text optional).

Output
- JSONL predictions.

Command
- poetry run verticalizer infer --in data/sites.csv --model models/us/v1/model.keras --calib models/us/v1/calib.pkl --out outputs/preds.jsonl --topk 26
