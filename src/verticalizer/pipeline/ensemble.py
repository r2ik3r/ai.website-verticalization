# src/verticalizer/pipeline/ensemble.py

from typing import List, Optional
import numpy as np
from ..models.persistence import loadmodel
from ..models.calibration import ProbCalibrator

def average_probs(prob_arrays: List[np.ndarray], weights: Optional[List[float]] = None, method: str = "mean") -> np.ndarray:
    if not prob_arrays:
        return np.zeros((0,0), dtype=np.float32)
    Xs = [np.asarray(p, dtype=np.float32) for p in prob_arrays]
    if method == "mean":
        if weights and len(weights) == len(Xs):
            w = np.asarray(weights, dtype=np.float32).reshape(-1, 1, 1)
            stacked = np.stack(Xs, axis=0)
            return np.sum(w * stacked, axis=0) / (np.sum(w) + 1e-9)
        else:
            return np.mean(np.stack(Xs, axis=0), axis=0)
    elif method == "softmax_mean":
        # Softmax on each model output row-wise, then mean
        def softmax(z):
            z = np.clip(z, -20, 20)
            e = np.exp(z)
            return e / (np.sum(e, axis=1, keepdims=True) + 1e-9)
        sm = [softmax(p) for p in Xs]
        return np.mean(np.stack(sm, axis=0), axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

def load_many_models(model_paths: List[str]):
    return [loadmodel(p) for p in model_paths]

def load_many_calibrators(calib_paths: List[str]):
    out: List[ProbCalibrator] = []
    for p in calib_paths:
        out.append(ProbCalibrator.load(p) if p else ProbCalibrator())
    return out

def apply_many_calibrators(raw_arrays: List[np.ndarray], calibrators: List[ProbCalibrator]) -> List[np.ndarray]:
    outs = []
    for raw, cal in zip(raw_arrays, calibrators):
        if getattr(cal, "cals", None):
            outs.append(cal.transform(raw))
        else:
            outs.append(raw)
    return outs