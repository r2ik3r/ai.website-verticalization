import pandas as pd
import json
from verticalizer.utils.taxonomy import load_taxonomy  # Our helper

def excel_to_training_csv(xlsx_path, geo, out_csv, out_json):
    df = pd.read_excel(xlsx_path)
    id2label, label2id = load_taxonomy()

    rows = []
    gold = {}

    for _, row in df.iterrows():
        site = str(row['website']).strip().lower()
        labels = []
        scores = {}
        for i in [1, 2, 3]:
            vcol = f'vertical{i}'
            if vcol in row and pd.notna(row[vcol]):
                label_str = str(row[vcol]).strip()
                if label_str in label2id:
                    iab_id = label2id[label_str]
                    labels.append(iab_id)
                    scores[iab_id] = int(row['score'])
                else:
                    print(f"[WARN] Label not in mapping: {label_str} for {site}")
        if labels:
            rows.append({
                "website": site,
                "iab_labels": json.dumps(labels),
                "premiumness_labels": json.dumps(scores),
                "geo": geo,
                "content_text": ""
            })
            gold[site] = scores

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(gold, f, indent=2)
    print(f"✅ Wrote {out_csv} and {out_json}")

excel_to_training_csv("US.xlsx", "US", "data/us_labeled.csv", "data/us_groundtruth.json")
excel_to_training_csv("India.xlsx", "IN", "data/in_labeled.csv", "data/in_groundtruth.json")
