import pandas as pd
from typing import Dict, Any
from ...pipeline.nodes import train_from_labeled
from ...models.registry import save_artifacts

def train_from_csv(labeled_csv: str, geo: str, version: str, out_base: str, config: Dict[str, Any] | None = None):
    df = pd.read_csv(labeled_csv)
    bundle = train_from_labeled(df)
    cfg = config or {"loss_weights": {"labels": 1.0, "scores": 0.3}}
    model_path, calib_path = save_artifacts(geo, version, bundle["model"], bundle["cal"], out_base, cfg)
    metrics = {"note": "Train complete; evaluate with a held-out set via the eval module."}
    return {"model": model_path, "calib": calib_path, "metrics": metrics}
