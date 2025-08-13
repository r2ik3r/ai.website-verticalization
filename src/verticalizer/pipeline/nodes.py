import numpy as np
import pandas as pd
import json
from typing import Dict, Any, List
from ..features.builder import crawl_and_embed
from ..models.keras_multilabel import build_model
from ..models.calibration import ProbCalibrator
from ..utils.taxonomy import load_taxonomy
from ..utils.logging import get_logger
from ..utils.metrics import multilabel_metrics, topk_accuracy

log = get_logger(__name__)

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
        labs = [l for l in labs if isinstance(l, str) and l.upper().startswith("IAB")]
        for lab in labs:
            if lab in idx:
                y_labels[r, idx[lab]] = 1.0

        # Scores
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
                sval = max(1.0, min(10.0, sval))
                y_scores[r, idx[lab]] = sval / 10.0

    return {
        "labels": y_labels.astype(np.float32),
        "scores": y_scores.astype(np.float32)
    }

def train_from_labeled(df: pd.DataFrame) -> Dict[str, Any]:
    """Train classifier + score regressor from labeled DataFrame."""
    log.info(f"[TRAIN] Starting with {len(df)} samples")
    id2label, _ = load_taxonomy()
    classes = list(id2label.keys())

    # Crawl + embed
    df = crawl_and_embed(df)
    if "embedding" not in df.columns or len(df) == 0:
        raise RuntimeError("No embeddings found! Check crawl/embedding.")
    X = np.stack(df["embedding"].values).astype(np.float32)

    # Prepare targets
    tgt = _prepare_targets(df, classes)
    y_labels, y_scores = tgt["labels"], tgt["scores"]

    log.info(f"[TRAIN] Shapes: X={X.shape}, y_labels={y_labels.shape}, y_scores={y_scores.shape}")
    log.info(f"[TRAIN] Nonzero label count: {y_labels.sum()}")

    # Build and train model
    model = build_model(X.shape[1], len(classes))
    log.info(f"[TRAIN] Starting keras fit: X={X.shape}, y_labels={y_labels.sum()} total positive labels")
    model.fit(X, {"labels": y_labels, "scores": y_scores},
              validation_split=0.2, epochs=12, batch_size=32, verbose=2)

    # Calibrate classification head
    val_ix = np.arange(len(X)) % 5 == 0
    cal = ProbCalibrator()
    if val_ix.sum() >= 5:
        p_val, _ = model.predict(X[val_ix], verbose=0)
        cal.fit(p_val, y_labels[val_ix])
        log.info("[TRAIN] Calibration fitted on holdout set")
    else:
        log.warning("[TRAIN] Skipping calibration (too few holdout samples)")

    return {"model": model, "cal": cal, "classes": classes}

def infer(model, cal: ProbCalibrator, classes: List[str], df: pd.DataFrame, topk: int = 3):
    """Run inference and return top-k predictions per site."""
    log.info(f"[INFER] On {len(df)} samples, topk={topk}")
    from ..utils.taxonomy import load_taxonomy
    id2label, _ = load_taxonomy()

    df = crawl_and_embed(df)
    X = np.stack(df["embedding"].values).astype(np.float32)

    raw_probs, scores = model.predict(X, verbose=0)
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
    df = crawl_and_embed(df)
    X = np.stack(df["embedding"].values).astype(np.float32)

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
                labs = [x.strip() for x in raw.split(",")]
        elif isinstance(raw, list):
            labs = raw
        labs = [l for l in labs if l.upper().startswith("IAB")]
        for lab in labs:
            if lab in idx:
                Y[i, idx[lab]] = 1.0

    raw_probs, _ = model.predict(X, verbose=0)
    probs = cal.transform(raw_probs) if getattr(cal, "cals", None) else raw_probs
    m = multilabel_metrics(Y, probs, threshold=0.5)
    m["top1_acc"] = topk_accuracy(Y, probs, k=1)
    m["top3_acc"] = topk_accuracy(Y, probs, k=3)
    return m
