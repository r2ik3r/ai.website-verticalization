# src/verticalizer/apps/trainer/service.py

import pandas as pd
from typing import Dict, Any
from ...pipeline.nodes import train_from_labeled
from ...models.registry import save_artifacts

def train_from_csv(labeled_csv: str, geo: str, version: str, out_base: str, config: Dict[str, Any] = None):
    df = pd.read_csv(labeled_csv)
    bundle = train_from_labeled(df, cfg=config or {})
    model_path, calib_path = save_artifacts(geo, version, bundle["model"], bundle["cal"], out_base, config or {})
    return {"model": model_path, "calib": calib_path, "metrics": bundle["metrics"]}