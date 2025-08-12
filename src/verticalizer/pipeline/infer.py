def run_infer(input_csv, model_path, calib_path, out_jsonl, geo="US", topk=26):
    from ..models.persistence import load_model
    from ..models.calibration import ProbCalibrator
    from ..utils.taxonomy import load_taxonomy
    from ..features.builder import crawl_and_embed
    from .io import read_table, write_jsonl
    import numpy as np

    id2label, _ = load_taxonomy()
    label_space = list(id2label.keys())
    model = load_model(model_path)
    calib = ProbCalibrator.load(calib_path)

    df = read_table(input_csv)
    df = crawl_and_embed(df)
    X = np.stack(df["embedding"].values)

    probs, scores = model.predict(X, verbose=0)
    probs = calib.transform(probs) if calib.cals else probs
    scores_int = np.clip((scores * 10).round().astype(int), 1, 10)

    rows = []
    for i, row in df.iterrows():
        order = np.argsort(-probs[i])[:topk]
        cats = [{
            "id": label_space[j],
            "label": id2label[label_space[j]],
            "prob": float(probs[i, j]),
            "score": int(scores_int[i, j])
        } for j in order]
        rows.append({
            "website": row["website"],
            "geo": geo,
            "categories": cats
        })
    write_jsonl(out_jsonl, rows)
