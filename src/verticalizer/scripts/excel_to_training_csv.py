import pandas as pd
import json

def excel_to_training_csv(xlsx_path: str, geo: str, out_csv: str, out_json: str):
    """
    Convert fixed-format Excel file into:
        - Training CSV for Verticalizer model
        - Groundtruth JSON for evaluation

    Always keeps every site in the CSV (Site is mandatory and never empty).

    If Premiumness Score is missing:
        - site still included
        - iab_labels (if present) retained for classification training
        - premiumness_labels left empty to allow model to learn/predict scores
    """

    df = pd.read_excel(xlsx_path, sheet_name=0)

    expected_cols = [
        "Site",
        "Vertical1 IAB",
        "Vertical2 IAB",
        "Vertical3 IAB",
        "Premiumness Score"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    rows = []
    gold = {}

    for _, row in df.iterrows():
        site = str(row["Site"]).strip().lower()  # always present

        labels = []
        scores = {}

        # Check if global Premiumness Score exists
        score_val = None
        if not pd.isna(row["Premiumness Score"]) and str(row["Premiumness Score"]).strip() != "":
            try:
                score_val = int(row["Premiumness Score"])
                score_val = max(1, min(10, score_val))
            except Exception:
                score_val = None

        # Loop over possible IAB columns
        for col in ["Vertical1 IAB", "Vertical2 IAB", "Vertical3 IAB"]:
            if pd.isna(row[col]) or not str(row[col]).strip():
                continue
            iab_id = str(row[col]).strip()
            if not iab_id.upper().startswith("IAB"):
                continue
            if iab_id not in labels:
                labels.append(iab_id)
                # Assign score only if global score is available
                if score_val is not None:
                    scores[iab_id] = score_val

        rows.append({
            "website": site,
            "iab_labels": json.dumps(labels),             # [] for no verticals
            "premiumness_labels": json.dumps(scores),     # {} if score unknown
            "geo": geo,
            "content_text": ""  # fetched at training if needed
        })
        gold[site] = scores

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2)

    print(f"✅ Wrote {out_csv} ({len(rows)} sites incl. unlabeled or score-missing)")
    print(f"✅ Wrote {out_json}")


if __name__ == "__main__":
    excel_to_training_csv(
        "src/verticalizer/data/input/SiteRank_US_US.xlsx",
        "US",
        "src/verticalizer/data/us_labeled.csv",
        "src/verticalizer/data/us_groundtruth.json"
    )
    excel_to_training_csv(
        "src/verticalizer/data/input/SiteRank_AS_IN.xlsx",
        "IN",
        "src/verticalizer/data/in_labeled.csv",
        "src/verticalizer/data/in_groundtruth.json"
    )
