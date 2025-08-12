import numpy as np
import pandas as pd
from typing import Tuple
from ..features.builder import crawl_and_embed
from ..models.keras_multilabel import build_model, to_bin_vector
from ..models.calibration import ProbCalibrator
from ..models.persistence import save_model
from ..utils.taxonomy import load_taxonomy, normalize_labels
from ..utils.metrics import multilabel_metrics, topk_accuracy

def prepare_labels(df: pd.DataFrame, id2label: dict, label2id: dict) -> Tuple[np.ndarray, list[str]]:
    classes = list(id2label.keys())
    idx = {c:i for i,c in enumerate(classes)}
    Y = np.zeros((len(df), len(classes)), dtype="float32")
    for i, row in df.iterrows():
        raw = row.get("iab_labels")
        if isinstance(raw, str):
            try:
                import json
                labels = json.loads(raw)
                if isinstance(labels, str):
                    labels = [labels]
            except Exception:
                labels = [x.strip() for x in raw.split(",")]
        elif isinstance(raw, list):
            labels = raw
        else:
            labels = []
        labels = normalize_labels(labels)
        for lab in labels:
            if lab in idx:
                Y[i, idx[lab]] = 1.0
    return Y, classes

def self_training_loop(seed_df: pd.DataFrame, unlabeled_df: pd.DataFrame, iterations: int = 3, min_conf: float = 0.8):
    id2label, label2id = load_taxonomy()
    # Step 1: crawl+embed both
    seed_df = crawl_and_embed(seed_df)
    unlabeled_df = crawl_and_embed(unlabeled_df)

    Y_seed, classes = prepare_labels(seed_df, id2label, label2id)
    emb_dim = len(seed_df.iloc[0]["embedding"]) if len(seed_df) else len(unlabeled_df.iloc[0]["embedding"])
    X_seed = np.stack(seed_df["embedding"].values)

    # Initialize model
    model = build_model(emb_dim=emb_dim, num_labels=len(classes))
    score_bins_seed = to_bin_vector([5]*len(seed_df))  # neutral placeholder unless provided

    model.fit(
        X_seed, {"labels": Y_seed, "score_bins": score_bins_seed},
        validation_split=0.1, epochs=10, batch_size=32, verbose=0
    )

    # Iterative pseudo-labeling
    for it in range(iterations):
        X_unl = np.stack(unlabeled_df["embedding"].values)
        probs, _ = model.predict(X_unl, verbose=0)
        # select high-confidence positives
        high = (probs >= min_conf).astype(int)
        any_high = high.sum(axis=1) > 0
        if any_high.sum() == 0:
            break
        X_sel = X_unl[any_high]
        Y_sel = high[any_high]
        score_bins_sel = to_bin_vector([6]*len(X_sel))
        # mix with existing
        X_train = np.concatenate([X_seed, X_sel], axis=0)
        Y_train = np.concatenate([Y_seed, Y_sel], axis=0)
        score_train = np.concatenate([score_bins_seed, score_bins_sel], axis=0)
        model.fit(
            X_train, {"labels": Y_train, "score_bins": score_train},
            validation_split=0.1, epochs=5, batch_size=32, verbose=0
        )
        # update seed set
        X_seed, Y_seed, score_bins_seed = X_train, Y_train, score_train

    # Calibration on last split of seed set
    val_ix = np.arange(len(X_seed)) % 5 == 0
    if val_ix.sum() > 10:
        val_probs, _ = model.predict(X_seed[val_ix], verbose=0)
        cal = ProbCalibrator()
        cal.fit(val_probs, Y_seed[val_ix])
    else:
        cal = ProbCalibrator()  # empty; identity transform during inference if not fitted

    return model, cal, classes
