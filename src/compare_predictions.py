import json

def compare_predictions(pred_jsonl, gold_json):
    gold = json.load(open(gold_json))
    preds = []
    with open(pred_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            preds.append(json.loads(line))
    for p in preds:
        site = p['website']
        if site in gold:
            gold_iabs = set(gold[site].keys())
            pred_iabs = {c['id'] for c in p['categories']}
            overlap = gold_iabs & pred_iabs
            print(f"{site}: Gold={gold_iabs}, Pred={pred_iabs}, Overlap={overlap}")

compare_predictions("outputs/us_predictions.jsonl", "data/us_groundtruth.json")
