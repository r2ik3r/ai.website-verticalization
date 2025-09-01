# src/verticalizer/apps/evaluate/service.py

import json
import numpy as np
from ...pipeline.io import readjsonl

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    # Simple ECE for multilabel: flatten positives/negatives
    y = y_true.flatten()
    p = y_prob.flatten()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (p >= b0) & (p < b1)
        if not np.any(m):
            continue
        conf = float(np.mean(p[m]))
        acc = float(np.mean(y[m]))
        ece += (np.sum(m) / len(p)) * abs(acc - conf)
    return float(ece)

def compare_jsonl_to_gold(pred_jsonl: str, gold_json: str, out_report: str):
    import orjson
    gold = orjson.loads(open(gold_json, "rb").read())
    preds = readjsonl(pred_jsonl)
    total = 0
    matched_any = 0
    details = []
    for p in preds:
        site = p.get("website")
        total += 1
        gold_iabs = set((gold.get(site) or {}).keys())
        pred_iabs = set([c["id"] for c in p.get("categories", [])])
        overlap = sorted(gold_iabs & pred_iabs)
        if overlap:
            matched_any += 1
        details.append({"website": site, "gold": sorted(gold_iabs), "pred": sorted(pred_iabs), "overlap": overlap})
    summary = {
        "sites": total,
        "matchedAny": matched_any,
        "matchRate": round(matched_any / total, 4) if total else 0.0,
    }
    report = {"summary": summary, "details": details}
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report