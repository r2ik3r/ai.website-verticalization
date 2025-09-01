# src/verticalizer/pipeline/drift.py

import json
from typing import Dict
import numpy as np
import pandas as pd
from .nodes import evaluate
from ..models.calibration import ProbCalibrator

def reembed_and_recalibrate(sample_csv: str, model, calib: ProbCalibrator, classes, out_report: str) -> Dict:
    """
    Re-embed a rolling sample (prepareembeddingsfordf inside evaluate) and optionally refit isotonic if drift detected.
    For simplicity, measure metrics and write a report; refitting decision left to operator.
    """
    df = pd.read_csv(sample_csv)
    metrics = evaluate(model, calib, classes, df)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics