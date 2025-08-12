import argparse
import os
import json
from .utils.logging import get_logger
from .utils.seed import seed_all
from .pipeline.io import read_table, write_jsonl
from .pipeline.nodes import train_from_labeled, infer as infer_nodes, evaluate as eval_nodes
from .models.persistence import save_model, load_model
from .models.calibration import ProbCalibrator
from .scripts.excel_to_training_csv import excel_to_training_csv

log = get_logger("verticalizer")


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def cmd_train(geo: str, in_path: str, model_out: str, calib_out: str, report: str):
    df = read_table(in_path)
    bundle = train_from_labeled(df)
    _ensure_dir(model_out)
    _ensure_dir(calib_out)
    _ensure_dir(report)
    save_model(bundle["model"], model_out)
    bundle["cal"].save(calib_out)
    # quick validation on same set (best to use a proper val split in practice)
    metrics = eval_nodes(bundle["model"], bundle["cal"], list(bundle["cal"].cals.keys()) or [], df)
    with open(report, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("Training completed")


def cmd_infer(geo: str, in_path: str, model: str, calib: str, out_path: str, topk: int):
    df = read_table(in_path)
    model_obj = load_model(model)
    cal = ProbCalibrator.load(calib) if calib and os.path.exists(calib) else ProbCalibrator()
    from .utils.taxonomy import load_taxonomy
    id2label, _ = load_taxonomy()
    classes = list(id2label.keys())
    out = infer_nodes(model_obj, cal, classes, df, topk=topk)
    _ensure_dir(out_path)
    write_jsonl(out_path, out)
    log.info("Inference completed")


def cmd_validate(geo: str, in_path: str, model: str, calib: str, report: str):
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
    log.info("Validation report written")


def cmd_self_train(geo: str, seed_path: str, unlabeled_path: str, iterations: int, model_out: str, calib_out: str, report: str):
    from .pipeline.self_train import self_training_loop
    seed_df = read_table(seed_path)
    unlabeled_df = read_table(unlabeled_path)
    model, cal, classes = self_training_loop(seed_df, unlabeled_df, iterations=iterations)
    _ensure_dir(model_out)
    _ensure_dir(calib_out)
    _ensure_dir(report)
    save_model(model, model_out)
    cal.save(calib_out)
    # Optional evaluation on seed set
    try:
        metrics = eval_nodes(model, cal, classes, seed_df)
    except Exception:
        metrics = {"note": "No labels available for seed evaluation"}
    with open(report, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("Self-training completed")


def compare_predictions(pred_jsonl: str, gold_json: str, out_report: str | None = None):
    import orjson
    gold = orjson.loads(open(gold_json, "rb").read())
    preds = []
    with open(pred_jsonl, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            preds.append(orjson.loads(line))
    # Compute simple overlap metrics
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
    summary = {
        "sites": total,
        "matched_any": matched_any,
        "match_rate": round(matched_any / total, 4) if total else 0.0,
    }
    report = {"summary": summary, "details": details}
    if out_report:
        _ensure_dir(out_report)
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


def cmd_run_pipeline(geo: str, excel_path: str, labeled_csv: str, groundtruth_json: str,
                     model_path: str, calib_path: str, output_path: str, report_path: str, topk: int = 26):
    if excel_to_training_csv is None:
        raise RuntimeError("scripts.excel_to_training_csv.excel_to_training_csv not available. Ensure PYTHONPATH=./src and the script exists.")
    # 1) Convert Excel → labeled CSV + ground truth
    excel_to_training_csv(excel_path, geo, labeled_csv, groundtruth_json)
    # 2) Train
    cmd_train(geo, labeled_csv, model_path, calib_path, report_path)
    # 3) Infer
    cmd_infer(geo, labeled_csv, model_path, calib_path, output_path, topk=topk)
    # 4) Compare predictions vs ground truth
    qa_report_path = os.path.splitext(report_path)[0] + ".compare.json"
    cmp_report = compare_predictions(output_path, groundtruth_json, qa_report_path)
    log.info(f"Compare report: {cmp_report['summary']}")


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

    # New integrated pipeline command
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
        cmd_self_train(args.geo, args.seed, args.unlabeled, args.iterations, args.model_out, args.calib_out, args.report)
    elif args.cmd == "run-pipeline":
        cmd_run_pipeline(args.geo, args.excel_path, args.labeled_csv, args.groundtruth_json,
                         args.model_path, args.calib_path, args.output_path, args.report_path, args.topk)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    app()


