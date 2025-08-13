# src/verticalizer/cli.py
import argparse
import os
import json
from .utils.logging import get_logger
from .utils.seed import seed_all
from .pipeline.io import read_table, write_jsonl
from .pipeline.nodes import train_from_labeled, infer as infer_nodes, evaluate as eval_nodes
from .models.persistence import save_model, load_model
from .models.calibration import ProbCalibrator

try:
    from .scripts.excel_to_training_csv import excel_to_training_csv
except Exception:
    excel_to_training_csv = None

log = get_logger("verticalizer")

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def cmd_train(geo, in_path, model_out, calib_out, report):
    log.info(f"[PIPE] Training model for geo={geo}")
    df = read_table(in_path)
    bundle = train_from_labeled(df)
    _ensure_dir(model_out); _ensure_dir(calib_out); _ensure_dir(report)
    save_model(bundle["model"], model_out)
    bundle["cal"].save(calib_out)
    metrics = eval_nodes(bundle["model"], bundle["cal"], bundle["classes"], df)
    with open(report, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"[PIPE] Training complete -> model={model_out}")

def cmd_infer(geo, in_path, model, calib, out_path, topk):
    log.info(f"[PIPE] Running inference geo={geo}, topk={topk}")
    df = read_table(in_path)
    model_obj = load_model(model)
    cal = ProbCalibrator.load(calib) if calib and os.path.exists(calib) else ProbCalibrator()
    from .utils.taxonomy import load_taxonomy
    id2label, _ = load_taxonomy()
    classes = list(id2label.keys())
    out = infer_nodes(model_obj, cal, classes, df, topk=topk)
    _ensure_dir(out_path)
    write_jsonl(out_path, out)
    log.info(f"[PIPE] Inference complete -> {out_path}")

def cmd_validate(geo: str, in_path: str, model: str, calib: str, report: str):
    log.info(f"[CLI] VALIDATE geo={geo} in={in_path}")
    df = read_table(in_path)
    model_obj = load_model(model)
    cal = ProbCalibrator.load(calib) if calib and os.path.exists(calib) else ProbCalibrator()
    from .utils.taxonomy import load_taxonomy
    id2label, _ = load_taxonomy()
    classes = list(id2label.keys())
    metrics = eval_nodes(model_obj, cal, classes, df)
    _ensure_dir(report)
    with open(report, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"[CLI] VALIDATE complete -> report={report}")

def compare_predictions(pred_jsonl: str, gold_json: str, out_report: str | None = None):
    import orjson
    gold = orjson.loads(open(gold_json, "rb").read())
    preds = []
    with open(pred_jsonl, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            preds.append(orjson.loads(line))
    total = 0
    matched_any = 0
    details = []
    for p in preds:
        site = p.get("website")
        total += 1
        gold_iabs = set((gold.get(site) or {}).keys())
        pred_iabs = set(c["id"] for c in p.get("categories", []))
        overlap = sorted(gold_iabs & pred_iabs)
        if overlap:
            matched_any += 1
        details.append({"website": site, "gold": sorted(gold_iabs), "pred": sorted(pred_iabs), "overlap": overlap})
    summary = {"sites": total, "matched_any": matched_any, "match_rate": round(matched_any / total, 4) if total else 0.0}
    report = {"summary": summary, "details": details}
    if out_report:
        _ensure_dir(out_report)
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report

def cmd_run_pipeline(geo, excel_path, labeled_csv, groundtruth_json, model_path, calib_path, out_path, report_path, topk):
    if excel_to_training_csv is None:
        raise RuntimeError("Excel converter not found in scripts.")
    log.info("[PIPE] Step 1: Converting Excel to CSV/JSON...")
    excel_to_training_csv(excel_path, geo, labeled_csv, groundtruth_json)
    log.info("[PIPE] Step 2: Training...")
    cmd_train(geo, labeled_csv, model_path, calib_path, report_path)
    log.info("[PIPE] Step 3: Inference...")
    cmd_infer(geo, labeled_csv, model_path, calib_path, out_path, topk)
    log.info("[PIPE] Step 4: Compare predictions with ground truth...")
    compare_path = os.path.splitext(report_path)[0] + ".compare.json"
    cmp = compare_predictions(out_path, groundtruth_json, compare_path)
    log.info(f"[PIPE] Compare summary: {cmp['summary']}")

def app():
    parser = argparse.ArgumentParser(prog="verticalizer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--geo", required=True)
    p_train.add_argument("--in", dest="in_path", required=True)
    p_train.add_argument("--model-out", required=True)
    p_train.add_argument("--calib-out", required=True)
    p_train.add_argument("--report", required=True)

    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--geo", required=True)
    p_infer.add_argument("--in", dest="in_path", required=True)
    p_infer.add_argument("--model", required=True)
    p_infer.add_argument("--calib", required=False, default=None)
    p_infer.add_argument("--out", required=True)
    p_infer.add_argument("--topk", type=int, default=3)

    p_eval = sub.add_parser("validate")
    p_eval.add_argument("--geo", required=True)
    p_eval.add_argument("--in", dest="in_path", required=True)
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--calib", required=False, default=None)
    p_eval.add_argument("--report", required=True)

    p_self = sub.add_parser("self-train")
    p_self.add_argument("--geo", required=True)
    p_self.add_argument("--seed", required=True)
    p_self.add_argument("--unlabeled", required=True)
    p_self.add_argument("--iterations", type=int, default=3)
    p_self.add_argument("--model-out", required=True)
    p_self.add_argument("--calib-out", required=True)
    p_self.add_argument("--report", required=True)

    p_pipe = sub.add_parser("run-pipeline")
    p_pipe.add_argument("--geo", required=True)
    p_pipe.add_argument("--excel-path", required=True)
    p_pipe.add_argument("--labeled-csv", required=True)
    p_pipe.add_argument("--groundtruth-json", required=True)
    p_pipe.add_argument("--model-path", required=True)
    p_pipe.add_argument("--calib-path", required=True)
    p_pipe.add_argument("--output-path", required=True)
    p_pipe.add_argument("--report-path", required=True)
    p_pipe.add_argument("--topk", type=int, default=26)

    args = parser.parse_args()
    seed_all(42)

    if args.cmd == "train":
        cmd_train(args.geo, args.in_path, args.model_out, args.calib_out, args.report)
    elif args.cmd == "infer":
        cmd_infer(args.geo, args.in_path, args.model, args.calib, args.out, args.topk)
    elif args.cmd == "validate":
        cmd_validate(args.geo, args.in_path, args.model, args.calib, args.report)
    elif args.cmd == "self-train":
        from .pipeline.self_train import self_training_loop
        log.info(f"[CLI] SELF-TRAIN geo={args.geo}")
        seed_df = read_table(args.seed)
        unlabeled_df = read_table(args.unlabeled)
        model, cal, classes = self_training_loop(seed_df, unlabeled_df, iterations=args.iterations)
        _ensure_dir(args.model_out); _ensure_dir(args.calib_out); _ensure_dir(args.report)
        save_model(model, args.model_out); cal.save(args.calib_out)
        try:
            metrics = eval_nodes(model, cal, classes, seed_df)
        except Exception:
            metrics = {"note": "No labels available for seed evaluation"}
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log.info("[CLI] SELF-TRAIN complete")
    elif args.cmd == "run-pipeline":
        cmd_run_pipeline(args.geo, args.excel_path, args.labeled_csv, args.groundtruth_json,
                         args.model_path, args.calib_path, args.output_path, args.report_path, args.topk)
    else:
        parser.error("Unknown command")

if __name__ == "__main__":
    app()
