import json
import os
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TAX_DIR = os.path.join(BASE_DIR, "data", "taxonomy")

def load_taxonomy() -> Tuple[Dict[str, str], Dict[str, str]]:
    with open(os.path.join(TAX_DIR, "iab_id_to_label.json"), "r", encoding="utf-8") as f:
        id2label = json.load(f)
    with open(os.path.join(TAX_DIR, "iab_label_to_id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    return id2label, label2id

def normalize_labels(raw_labels: List[str]) -> List[str]:
    id2label, label2id = load_taxonomy()
    out = []
    for x in raw_labels:
        x = x.strip()
        if x in id2label:
            out.append(x)
        elif x in label2id:
            out.append(label2id[x])
        else:
            # try lenient match
            k = x.lower()
            for lbl, i in label2id.items():
                if lbl.lower() == k:
                    out.append(i)
                    break
    return list(dict.fromkeys(out))  # dedupe, preserve order
