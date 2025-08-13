# Evaluation Module

Purpose
- Compare predictions vs ground truth and summarize overlap.

Input
- predictions JSONL, ground truth JSON.

Output
- eval JSON report with summary and details.

Command
- poetry run verticalizer eval --pred outputs/preds.jsonl --gold data/gt.json --out reports/eval.json
