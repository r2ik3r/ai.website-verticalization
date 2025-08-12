import pandas as pd
import json
from src.verticalizer.utils.taxonomy import load_taxonomy

def excel_to_training_csv(xlsx_path: str, geo: str, out_csv: str, out_json: str):
    """
    Convert a geo-specific Excel file into:
      1. Training CSV for the Verticalizer model
      2. Groundtruth JSON for evaluation

    Enrichment enhancement:
      - Uses both the summary sheet (top 3 verticals) and the detailed
        per-vertical scoring sheet to fill premiumness scores for ALL verticals
        that the site is mapped to, not just top 3.
    """
    # Load taxonomy
    id2label, label2id = load_taxonomy()

    # Read all sheets
    xls = pd.ExcelFile(xlsx_path)
    sheet_names = xls.sheet_names
    # Try to guess which are top3 summary vs detailed scoring
    summary_sheet = None
    detail_sheet = None
    for s in sheet_names:
        s_lower = s.lower()
        if "score" in s_lower or "ranking" in s_lower or "vertical" in s_lower:
            detail_sheet = s
        if "site" in s_lower or "summary" in s_lower or "top" in s_lower:
            summary_sheet = s
    # Default to first for summary, last for detail if not auto-detected
    if summary_sheet is None:
        summary_sheet = sheet_names[0]
    if detail_sheet is None:
        detail_sheet = sheet_names[-1]

    summary_df = pd.read_excel(xlsx_path, sheet_name=summary_sheet)
    detail_df = pd.read_excel(xlsx_path, sheet_name=detail_sheet)

    # Normalize headers
    summary_df.columns = [c.strip().lower() for c in summary_df.columns]
    detail_df.columns = [c.strip().lower() for c in detail_df.columns]

    rows = []
    gold = {}

    for _, row in summary_df.iterrows():
        site = str(row.get('site') or row.get('website') or "").strip().lower()
        if not site:
            continue

        labels = []
        scores = {}

        # ---- Step 1: Add top 3 from summary ----
        for i in [1, 2, 3]:
            v_name_col = f'vertical{i}'
            score_col = 'score'  # global score from summary
            if v_name_col in summary_df.columns and pd.notna(row[v_name_col]):
                lbl_str = str(row[v_name_col]).strip()
                # Map label to IAB ID
                if lbl_str in label2id:
                    iab_id = label2id[lbl_str]
                elif lbl_str in id2label:
                    iab_id = lbl_str
                else:
                    continue
                labels.append(iab_id)
                try:
                    score_val = int(row.get(score_col, 0))
                except Exception:
                    score_val = 0
                scores[iab_id] = max(1, min(10, score_val))

        # ---- Step 2: Enrich using detail/per-vertical sheet ----
        # Filter that site's rows in detail_df
        site_rows = detail_df[(detail_df.get('site') == site) | (detail_df.get('website') == site)]
        for _, dr in site_rows.iterrows():
            lbl_str = str(dr.get('vertical') or dr.get('category') or "").strip()
            if not lbl_str:
                continue
            if lbl_str in label2id:
                iab_id = label2id[lbl_str]
            elif lbl_str in id2label:
                iab_id = lbl_str
            else:
                continue
            try:
                score_val = int(dr.get('score', dr.get('premiumness', 0)))
            except Exception:
                score_val = 0
            score_val = max(1, min(10, score_val))
            if iab_id not in labels:
                labels.append(iab_id)
            scores[iab_id] = score_val

        # Deduplicate labels preserving order
        seen = set()
        labels = [x for x in labels if not (x in seen or seen.add(x))]
        if not labels:
            continue

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

    print(f"✅ Wrote {out_csv} (training CSV) with {len(rows)} rows")
    print(f"✅ Wrote {out_json} (groundtruth JSON)")

if __name__ == "__main__":
    # Adjust input/output paths as needed
    excel_to_training_csv(
        "./src/scripts/data/SiteRank_US_US.xlsx",
        "US",
        "./src/verticalizer/data/us_labeled.csv",
        "./src/verticalizer/data/us_groundtruth.json"
    )
    excel_to_training_csv(
        "./src/scripts/data/SiteRank_AS_IN.xlsx",
        "IN",
        "./src/verticalizer/data/in_labeled.csv",
        "./src/verticalizer/data/in_groundtruth.json"
    )

# PYTHONPATH=./src poetry run python -m src.scripts.excel_to_training_csv