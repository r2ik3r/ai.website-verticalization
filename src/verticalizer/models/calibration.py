# src/verticalizer/models/calibration.py
import numpy as np
from sklearn.isotonic import IsotonicRegression
import joblib

class ProbCalibrator:
    def __init__(self):
        self.cals = {}   # idx -> IsotonicRegression

    def fit(self, rawprobs: np.ndarray, ytrue: np.ndarray):
        L = rawprobs.shape[2]
        for i in range(L):
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(rawprobs[:, i], ytrue[:, i])
            self.cals[i] = cal

    def transform(self, rawprobs: np.ndarray) -> np.ndarray:
        out = np.zeros_like(rawprobs)
        L = rawprobs.shape[2]
        for i in range(L):
            if i in self.cals:
                out[:, i] = self.cals[i].transform(rawprobs[:, i])
            else:
                out[:, i] = rawprobs[:, i]
        return out

    def save(self, path: str):
        joblib.dump(self.cals, path)

    @staticmethod
    def load(path: str):
        obj = ProbCalibrator()
        obj.cals = joblib.load(path)
        return obj