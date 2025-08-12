import numpy as np
import pandas as pd
from typing import Dict, Any, List
from ..features.builder import crawl_and_embed
from ..models.keras_multilabel import build_model, to_bin_vector
from ..models.calibration import ProbCalibrator
from ..models.persistence import save_model, load_model
from ..utils.taxonomy import load_taxonomy, normalize_labels
from ..utils.metrics import multilabel_metrics, topk_accuracy

def train_from_labeled(df: pd.DataFrame) -> Dict[str, Any]:
    id2label, label2id = load_taxonomy()
    df = crawl_and_embed(df)
    classes = list(id2label.keys())
    idx = {c:i for i,c in enumerate(classes)}
    Y = np.zeros((len(df), len(classes)), dtype="float32")
    for i, row in df.iterrows():
        raw = row.get("iab_labels")
        labels = []
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
        labels = normalize_labels(labels)
        for lab in labels:
            if lab in idx:
                Y[i, idx[lab]] = 1.0
    X = np.stack(df["embedding"].values)
    emb_dim = len(df.iloc[0]["embedding"])
    model = build_model(emb_dim=emb_dim, num_labels=len(classes))
    # Score bins: use provided or neutral 5
    if "premiumness_labels" in df.columns:
        import json
        scores = []
        for s in df["premiumness_labels"].fillna("{}"):
            try:
                m = json.loads(s) if isinstance(s, str) else s
            except Exception:
                m = {}
            # choose primary if exists
            primary_score = 5
            if m:
                try:
                    primary_score = int(sorted(m.values(), reverse=True)[0])
                except Exception:
                    primary_score = 5
            scores.append(primary_score)
    else:
        scores = [5]*len(df)
    score_bins = to_bin_vector(scores, bins=10)

    model.fit(
        X, {"labels": Y, "score_bins": score_bins},
        validation_split=0.1,
        epochs=12,
        batch_size=32,
        verbose=0
    )

    # Calibration
    val_ix = np.arange(len(X)) % 5 == 0
    cal = ProbCalibrator()
    if val_ix.sum() > 10:
        val_probs, _ = model.predict(X[val_ix], verbose=0)
        cal.fit(val_probs, Y[val_ix])
    return {"model": model, "cal": cal, "classes": classes}

def infer(model, cal: ProbCalibrator, classes: List[str], df: pd.DataFrame, topk: int = 3):
    id2label, _ = load_taxonomy()
    df = crawl_and_embed(df)
    X = np.stack(df["embedding"].values)
    raw_probs, score_bins = model.predict(X, verbose=0)
    probs = cal.transform(raw_probs) if cal.cals else raw_probs
    out = []
    import numpy as np
    for i, row in df.iterrows():
        p = probs[i]
        order = np.argsort(-p)[:topk]
        cats = []
        for j in order:
            iid = classes[j]
            cats.append({
                "id": iid,
                "label": id2label.get(iid, iid),
                "prob": float(p[j]),
                # map score bins to 1-10 by argmax
                "score": int(np.argmax(score_bins[i]) + 1)
            })
        out.append({"website": row["website"], "categories": cats})
    return out

def evaluate(model, cal: ProbCalibrator, classes: List[str], df: pd.DataFrame) -> Dict[str, Any]:
    id2label, label2id = load_taxonomy()
    df = crawl_and_embed(df)
    X = np.stack(df["embedding"].values)
    # Build Y
    idx = {c:i for i,c in enumerate(classes)}
    import numpy as np
    Y = np.zeros((len(df), len(classes)), dtype="float32")
    for i, row in df.iterrows():
        raw = row.get("iab_labels")
        labels = []
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
        labels = normalize_labels(labels)
        for lab in labels:
            if lab in idx:
                Y[i, idx[lab]] = 1.0
    raw_probs, _ = model.predict(X, verbose=0)
    probs = cal.transform(raw_probs) if cal.cals else raw_probs
    from ..utils.metrics import multilabel_metrics, topk_accuracy
    m = multilabel_metrics(Y, probs, threshold=0.5)
    m["top1_acc"] = topk_accuracy(Y, probs, k=1)
    m["top3_acc"] = topk_accuracy(Y, probs, k=3)
    return m
