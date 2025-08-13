import os
from .persistence import save_model
from .calibration import ProbCalibrator
from ..storage.repositories import save_model_version

def save_artifacts(geo: str, version: str, model, calibrator: ProbCalibrator, base_dir: str, config: dict):
    model_dir = os.path.join(base_dir, geo, version)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.keras")
    calib_path = os.path.join(model_dir, "calib.pkl")
    save_model(model, model_path)
    calibrator.save(calib_path)
    save_model_version(geo, version, model_path, calib_path, config)
    return model_path, calib_path
