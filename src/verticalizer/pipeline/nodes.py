# src/verticalizer/pipeline/nodes.py

import json
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .common import prepare_embeddings_for_df
from ..models.calibration import ProbCalibrator
from ..models.keras_multilabel import build_model
from ..utils.metrics import multilabel_metrics, topk_accuracy
from ..utils.taxonomy import load_taxonomy

logger = logging.getLogger(__name__)


def _prepare_targets(df: pd.DataFrame, classes: List[str]) -> Dict[str, np.ndarray]:
    """Build float32 matrices for classification labels and vertical scores."""
    idx = {c: i for i, c in enumerate(classes)}
    n, L = len(df), len(classes)
    y_labels = np.zeros((n, L), dtype=np.float32)
    y_scores = np.zeros((n, L), dtype=np.float32)

    for r, row in df.iterrows():
        # Classification labels
        labs = []
        raw_labs = row.get("iab_labels")
        if isinstance(raw_labs, str) and raw_labs.strip():
            try:
                labs = json.loads(raw_labs)
                if isinstance(labs, str):
                    labs = [labs]
            except Exception:
                labs = [x.strip() for x in raw_labs.split(",") if x.strip()]
        elif isinstance(raw_labs, list):
            labs = raw_labs
        # Only keep valid IAB IDs
        labs = [label for label in labs if isinstance(label, str) and label.upper().startswith("IAB")]
        for lab in labs:
            if lab in idx:
                y_labels[r, idx[lab]] = 1.0

        # Scores (premium scores)
        score_map = {}
        raw_scores = row.get("premiumness_labels")
        if isinstance(raw_scores, str) and raw_scores.strip():
            try:
                score_map = json.loads(raw_scores)
            except Exception:
                score_map = {}
        elif isinstance(raw_scores, dict):
            score_map = raw_scores

        for lab, s in score_map.items():
            if lab in idx:
                try:
                    sval = float(s)
                except Exception:
                    sval = 0.0
                # clamp 1..10 then normalize to 0..1
                sval = max(1.0, min(10.0, sval))
                y_scores[r, idx[lab]] = sval / 10.0

    return {
        "labels": y_labels.astype(np.float32),
        "scores": y_scores.astype(np.float32)
    }


def train_from_labeled(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Full training routine:
    1. Prepare embeddings via shared helper (crawl + embed + load from DB/cache)
    2. Train two-head model (labels + premiumness scores)
    3. Calibrate (if enough data)
    """
    logger.info(f"[TRAIN] Starting with {len(df)} samples")

    # Step 1-3: prepare embeddings using shared helper
    X = prepare_embeddings_for_df(df)

    # Step 4: Build label/score matrices
    id2label, _ = load_taxonomy()
    classes = list(id2label.keys())
    targets = _prepare_targets(df, classes)
    y_labels, y_scores = targets["labels"], targets["scores"]

    # Step 5: Create & fit model
    model = build_model(
        emb_dim=X.shape[1],
        num_labels=len(classes),
        hidden=512,
        dropout=0.3
    )
    logger.info("[TRAIN] Starting keras fit")
    model.fit(
        X,
        {"labels": y_labels, "scores": y_scores},
        validation_split=0.2,
        epochs=12,
        batch_size=32,
        verbose=2
    )

    # Step 6: Calibrate
    cal = ProbCalibrator()
    if np.sum(y_labels) > 5:  # only if enough positive labels
        pred_out = model.predict(X, verbose=0)
        raw_probs = pred_out if isinstance(pred_out, (list, tuple)) else pred_out
        if isinstance(raw_probs, (list, tuple)):
            raw_probs = raw_probs[0]  # labels head
        cal.fit(raw_probs, y_labels)

    return {"model": model, "cal": cal, "classes": classes}


def infer(model, cal: ProbCalibrator, classes: List[str], df: pd.DataFrame, topk: int = 3):
    """Run inference and return top‑k predictions per site."""
    logger.info(f"[INFER] On {len(df)} samples, topk={topk}")
    id2label, _ = load_taxonomy()

    X = prepare_embeddings_for_df(df)
    raw_probs, scores = model.predict(X, verbose=0)

    # if calibration is available, calibrate classification head
    if isinstance(raw_probs, (list, tuple)):
        raw_probs = raw_probs[0]
    probs = cal.transform(raw_probs) if getattr(cal, "cals", None) else raw_probs

    scores_int = np.clip((scores * 10).round().astype(int), 1, 10)

    out = []
    for i, row in df.iterrows():
        order = np.argsort(-probs[i])[:topk]
        cats = [{
            "id": classes[j],
            "label": id2label.get(classes[j], classes[j]),
            "prob": float(probs[i, j]),
            "score": int(scores_int[i, j])
        } for j in order]
        out.append({"website": row["website"], "categories": cats})
    return out


def evaluate(model, cal: ProbCalibrator, classes: List[str], df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate classification performance."""
    X = prepare_embeddings_for_df(df)

    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(df), len(classes)), dtype=np.float32)
    for i, row in df.iterrows():
        labs = []
        raw = row.get("iab_labels")
        if isinstance(raw, str) and raw.strip():
            try:
                labs = json.loads(raw)
                if isinstance(labs, str):
                    labs = [labs]
            except Exception:
                labs = [x.strip() for x in raw.split(",") if x.strip()]
        elif isinstance(raw, list):
            labs = raw
        labs = [label for label in labs if isinstance(label, str) and label.upper().startswith("IAB")]
        for lab in labs:
            if lab in idx:
                Y[i, idx[lab]] = 1.0

    raw_probs, _ = model.predict(X, verbose=0)
    if isinstance(raw_probs, (list, tuple)):
        raw_probs = raw_probs[0]
    probs = cal.transform(raw_probs) if getattr(cal, "cals", None) else raw_probs

    m = multilabel_metrics(Y, probs, threshold=0.5)
    m["top1_acc"] = topk_accuracy(Y, probs, k=1)
    m["top3_acc"] = topk_accuracy(Y, probs, k=3)
    return m
