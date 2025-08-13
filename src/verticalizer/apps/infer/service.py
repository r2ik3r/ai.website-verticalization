# src/verticalizer/apps/infer/service.py
import os
import pandas as pd
from ...models.persistence import load_model
from ...models.calibration import ProbCalibrator
from ...utils.taxonomy import load_taxonomy
from ...pipeline.nodes import infer as infer_nodes
from ...pipeline.io import write_jsonl

def infer_from_csv(in_csv: str, model_path: str, calib_path: str, out_jsonl: str, topk: int = 10):
    df = pd.read_csv(in_csv)
    cal = ProbCalibrator.load(calib_path) if calib_path and os.path.exists(calib_path) else ProbCalibrator()
    id2label, _ = load_taxonomy()
    classes = list(id2label.keys())
    model_obj = load_model(model_path)

    results = infer_nodes(model_obj, cal, classes, df, topk=topk)
    write_jsonl(out_jsonl, results)
    return out_jsonl
