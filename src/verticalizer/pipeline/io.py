from typing import List, Dict, Any
import orjson
import pandas as pd
from datetime import datetime

def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError("Unsupported table format")

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r))
            f.write(b"\n")

def now_iso():
    return datetime.utcnow().isoformat() + "Z"
