import numpy as np
import json
from .io import read_table
from ..models.keras_multilabel import build_model
from ..features.builder import crawl_and_embed
from ..utils.taxonomy import load_taxonomy

def prepare_label_and_score_targets(df, all_ids):
    idx = {cat: i for i, cat in enumerate(all_ids)}
    y_labels = np.zeros((len(df), len(all_ids)), dtype=np.float32)
    y_scores = np.zeros((len(df), len(all_ids)), dtype=np.float32)
    for r, row in df.iterrows():
        # Parse labels
        try:
            labs = json.loads(row["iab_labels"]) if isinstance(row["iab_labels"], str) else row["iab_labels"]
        except Exception:
            labs = []
        # Parse scores
        try:
            scores = json.loads(row["premiumness_labels"]) if isinstance(row["premiumness_labels"], str) else row["premiumness_labels"]
        except Exception:
            scores = {}
        for lab in (labs or []):
            if lab in idx:
                y_labels[r, idx[lab]] = 1.0
        for lab, score in (scores or {}).items():
            if lab in idx:
                y_scores[r, idx[lab]] = max(1, min(10, int(score))) / 10.0
    return y_labels, y_scores

def run_train(labeled_csv, model_out, calib_out):
    df = read_table(labeled_csv)
    id2label, _ = load_taxonomy()
    label_space = list(id2label.keys())

    df = crawl_and_embed(df)
    y_labels, y_scores = prepare_label_and_score_targets(df, label_space)
    X = np.stack(df["embedding"].values)

    model = build_model(X.shape[1], len(label_space))
    model.fit(X, {"labels": y_labels, "scores": y_scores},
              validation_split=0.2, epochs=12, batch_size=32, verbose=2)

    # Calibrate classification probabilities
    from ..models.calibration import ProbCalibrator
    val_idx = np.arange(len(X)) % 5 == 0
    calib = ProbCalibrator()
    if val_idx.sum() > 5:
        p_val, _ = model.predict(X[val_idx], verbose=0)
        calib.fit(p_val, y_labels[val_idx])
    calib.save(calib_out)

    from ..models.persistence import save_model
    save_model(model, model_out)
