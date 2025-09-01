# src/verticalizer/models/calibration.py
import numpy as np
from sklearn.isotonic import IsotonicRegression
import joblib

class ProbCalibrator:
    def __init__(self):
        self.cals = {}

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray):
        L = raw_probs.shape[1]
        for i in range(L):
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw_probs[:, i], y_true[:, i])
            self.cals[i] = cal

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        out = np.zeros_like(raw_probs)
        for i, cal in self.cals.items():
            out[:, i] = cal.transform(raw_probs[:, i])
        return out

    def save(self, path: str):
        joblib.dump(self.cals, path)

    @staticmethod
    def load(path: str):
        obj = ProbCalibrator()
        obj.cals = joblib.load(path)
        return obj
