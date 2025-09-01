# src/verticalizer/scripts/ingest_kaggle_iab.py

import json
import os
import re
from typing import List, Tuple
import pandas as pd
from ..utils.taxonomy_versioned import normalize_labels

def infer_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url if url.startswith("http") else f"https://{url}").netloc
        return netloc.lower().strip()
    except Exception:
        return (url or "").lower().strip()

def row_to_record(row, label_cols: List[str], text_cols: List[str], version: str) -> Tuple[str, List[str], str]:
    labs = []
    for c in label_cols:
        v = str(row.get(c, "") or "").strip()
        if not v:
            continue
        labs.append(v)
    labs = normalize_labels(labs, version=version)
    texts = []
    for c in text_cols:
        v = str(row.get(c, "") or "")
        if v and v.strip():
            texts.append(v.strip())
    content = "\n".join(texts)
    # Basic cleanup
    content = re.sub(r"\s+", " ", content).strip()
    website = infer_domain(row.get("url") or row.get("URL") or row.get("site") or row.get("domain") or "")
    return website, labs, content

def ingest_kaggle(
    kaggle_csv: str,
    out_csv: str,
    iab_version: str = "v3",
    label_cols: List[str] = ("Tier1","Tier2","Tier3","Tier4"),
    text_cols: List[str] = ("title","description","content","text"),
    min_labels: int = 1,
):
    """
    Reads a Kaggle IAB dataset CSV and writes repository CSV schema:
      website,iablabels,contenttext
    """
    df = pd.read_csv(kaggle_csv)
    rows = []
    seen = set()
    for _, r in df.iterrows():
        website, labs, content = row_to_record(r, list(label_cols), list(text_cols), iab_version)
        if not website or len(labs) < min_labels or not content:
            continue
        key = (website, hash(content))
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "website": website,
            "iablabels": json.dumps(labs, ensure_ascii=False),
            "contenttext": content,
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(out_df)} rows")
    
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--iab-version", default="v3")
    ap.add_argument("--label-cols", nargs="*", default=("Tier1","Tier2","Tier3"))
    ap.add_argument("--text-cols", nargs="*", default=("title","description","content"))
    ap.add_argument("--min-labels", type=int, default=1)
    args = ap.parse_args()
    ingest_kaggle(args.kaggle_csv, args.out_csv, args.iab_version, args.label_cols, args.text_cols, args.min_labels)