# src/verticalizer/pipeline/nodes.py

import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from .common import prepareembeddingsfordf
from ..models.calibration import ProbCalibrator
from ..models.kerasmultilabel import build_model
from ..utils.metrics import multilabelmetrics, topkaccuracy
from ..utils.taxonomy_versioned import load_taxonomy

logger = logging.getLogger(__name__)

def _parse_iab_list(raw):
    if isinstance(raw, list):
        labs = [str(x).strip() for x in raw]
    elif isinstance(raw, str) and raw.strip():
        try:
            v = json.loads(raw)
            if isinstance(v, list):
                labs = [str(x).strip() for x in v]
            else:
                labs = [x.strip() for x in raw.split(",") if x.strip()]
        except Exception:
            labs = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        labs = []
    return [x for x in labs if x.upper().startswith("IAB")]

def _prepare_targets(df: pd.DataFrame, classes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    idx = {c: i for i, c in enumerate(classes)}
    n, L = len(df), len(classes)
    ylabels = np.zeros((n, L), dtype=np.float32)
    yscores = np.zeros((n, L), dtype=np.float32)
    for r, row in df.iterrows():
        labs = _parse_iab_list(row.get("iablabels"))
        for lab in labs:
            if lab in idx:
                ylabels[r, idx[lab]] = 1.0
        # Optional premiumness map by IAB
        rawscores = row.get("premiumnesslabels")
        scoremap = None
        if isinstance(rawscores, str) and rawscores.strip():
            try:
                scoremap = json.loads(rawscores)
            except Exception:
                scoremap = None
        elif isinstance(rawscores, dict):
            scoremap = rawscores
        if scoremap:
            for lab, s in scoremap.items():
                if lab in idx:
                    try:
                        sval = float(s)
                    except Exception:
                        sval = 0.0
                    yscores[r, idx[lab]] = max(1.0, min(10.0, sval)) / 10.0
    return ylabels, yscores

def train_from_labeled(df: pd.DataFrame, cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = cfg or {}
    id2label, _, _, _ = load_taxonomy(cfg.get("iab_version", "v3"))
    classes = list(id2label.keys())
    X = prepareembeddingsfordf(df)
    ylabels, yscores = _prepare_targets(df, classes)

    model = build_model(
        embdim=int(X.shape[2]),
        numlabels=len(classes),
        hidden=int(cfg.get("hidden", 512)),
        dropout=float(cfg.get("dropout", 0.3)),
        labels_loss=str(cfg.get("labels_loss", "bce")),
        gamma=float(cfg.get("gamma", 2.0)),
    )
    callbacks = []
    if bool(cfg.get("early_stop", True)):
        callbacks.append(
            __import__("tensorflow").keras.callbacks.EarlyStopping(
                monitor="val_labels_auc", mode="max", patience=int(cfg.get("patience", 3)), restore_best_weights=True
            )
        )
    logger.info("TRAIN starting fit")
    model.fit(
        X,
        {"labels": ylabels, "scores": yscores},
        validation_split=float(cfg.get("val_split", 0.2)),
        epochs=int(cfg.get("epochs", 15)),
        batch_size=int(cfg.get("batch_size", 64)),
        verbose=2,
        callbacks=callbacks,
    )
    calibrator = ProbCalibrator()
    rawprobs = model.predict(X, verbose=0)
    if isinstance(rawprobs, (list, tuple)):
        rawprobs = rawprobs
    poscounts = ylabels.sum(axis=0)
    mask = poscounts >= 5.0
    if mask.any():
        calibrator.fit(rawprobs[:, mask], ylabels[:, mask])  # fit only where sufficient positives

    metrics = evaluate(model, calibrator, classes, df)
    return {"model": model, "cal": calibrator, "classes": classes, "metrics": metrics}

def infer(model, cal: ProbCalibrator, classes: List[str], df: pd.DataFrame, topk: int = 10) -> List[Dict[str, Any]]:
    X = prepareembeddingsfordf(df)
    raw = model.predict(X, verbose=0)
    if isinstance(raw, (list, tuple)):
        raw = raw
    probs = cal.transform(raw) if getattr(cal, "cals", None) else raw
    id2label, _, _, _ = load_taxonomy()
    out = []
    for i, row in enumerate(df.itertuples(index=False)):
        order = np.argsort(-probs[i])[:max(1, topk)]
        cats = [{"id": classes[j], "label": id2label.get(classes[j], classes[j]), "prob": float(probs[i, j])} for j in order]
        out.append({"website": str(getattr(row, "website")).strip().lower(), "categories": cats})
    return out

def evaluate(model, calibrator: ProbCalibrator, classes: List[str], df: pd.DataFrame) -> Dict[str, Any]:
    # Build Y from df
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(df), len(classes)), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        for lab in _parse_iab_list(getattr(row, "iablabels")):
            if lab in idx:
                Y[i, idx[lab]] = 1.0
    X = prepareembeddingsfordf(df)
    raw = model.predict(X, verbose=0)
    if isinstance(raw, (list, tuple)):
        raw = raw
    probs = calibrator.transform(raw) if getattr(calibrator, "cals", None) else raw
    m = multilabelmetrics(Y, probs, threshold=0.5)
    m["top1"] = topkaccuracy(Y, probs, k=1)
    m["top3"] = topkaccuracy(Y, probs, k=3)
    return m