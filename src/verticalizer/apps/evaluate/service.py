import json
from ...pipeline.io import read_jsonl

def compare_jsonl_to_gold(pred_jsonl: str, gold_json: str, out_report: str):
    import orjson
    gold = orjson.loads(open(gold_json, "rb").read())
    preds = read_jsonl(pred_jsonl)
    total = 0; matched_any = 0; details = []
    for p in preds:
        site = p.get("website")
        total += 1
        gold_iabs = set((gold.get(site) or {}).keys())
        pred_iabs = set(c["id"] for c in p.get("categories", []))
        overlap = sorted(gold_iabs & pred_iabs)
        if overlap:
            matched_any += 1
        details.append({"website": site, "gold": sorted(gold_iabs), "pred": sorted(pred_iabs), "overlap": overlap})
    summary = {"sites": total, "matched_any": matched_any, "match_rate": round(matched_any/total, 4) if total else 0.0}
    report = {"summary": summary, "details": details}
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
