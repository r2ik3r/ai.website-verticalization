# src/verticalizer/scripts/excel_to_training_csv.py
import pandas as pd
import json

def excel_to_training_csv(xlsx_path: str, geo: str, out_csv: str, out_json: str):
    df = pd.read_excel(xlsx_path, sheet_name=0)
    expected_cols = ["Site", "Vertical1 IAB", "Vertical2 IAB", "Vertical3 IAB", "Premiumness Score"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    rows = []
    gold = {}
    for _, row in df.iterrows():
        site = str(row["Site"]).strip().lower()
        if not site:
            continue

        labels = []
        for col in ("Vertical1 IAB", "Vertical2 IAB", "Vertical3 IAB"):
            v = row.get(col)
            if pd.isna(v):
                continue
            iabid = str(v).strip().upper()
            if iabid.startswith("IAB") and iabid not in labels:
                labels.append(iabid)

        scoreval = None
        if not pd.isna(row.get("Premiumness Score")) and str(row["Premiumness Score"]).strip():
            try:
                scoreval = int(row["Premiumness Score"])
                scoreval = max(1, min(10, scoreval))
            except Exception:
                scoreval = None

        scores = {}
        if scoreval is not None:
            for iabid in labels:
                scores[iabid] = scoreval

        rows.append({
            "website": site,
            "iablabels": json.dumps(labels),  # keep IDs
            "premiumnesslabels": json.dumps(scores),  # may be {}
            "geo": geo,
        })
        if labels:
            gold[site] = {iab: 1 for iab in labels}

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2)
    print(f"Wrote {out_csv} ({len(rows)} rows)")
    print(f"Wrote {out_json}")


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
