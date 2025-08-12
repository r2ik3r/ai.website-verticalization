# src/verticalizer/pipeline/infer.py
from ..models.persistence import load_model
from ..models.calibration import ProbCalibrator
from ..utils.taxonomy import load_taxonomy
from ..features.builder import crawl_and_embed
from .io import read_table
import numpy as np, json

def run_infer(input_csv, model_path, calib_path, out_jsonl, geo="US", topk=26):
    id2label, _ = load_taxonomy()
    label_space = list(id2label.keys())
    model = load_model(model_path)
    calib = ProbCalibrator.load(calib_path)

    df = read_table(input_csv)
    df = crawl_and_embed(df)
    X = np.stack(df["embedding"].values)

    probs, scores = model.predict(X, verbose=0)  # both num_labels
    probs = calib.transform(probs) if calib.cals else probs
    scores_int = np.clip((scores * 10).round().astype(int), 1, 10)

    rows = []
    for i, row in df.iterrows():
        p = probs[i]
        s = scores_int[i]
        order = np.argsort(-p)[:topk]
        cats = []
        for j in order:
            cats.append({
                "id": label_space[j],
                "label": id2label[label_space[j]],
                "prob": float(p[j]),
                "score": int(s[j])  # already 1–10
            })
        rows.append({
            "website": row["website"],
            "geo": geo,
            "categories": cats
        })

    from .io import write_jsonl
    write_jsonl(out_jsonl, rows)
