import argparse
import os
import json
from .utils.logging import get_logger
from .utils.seed import seed_all
from .pipeline.io import read_table, write_jsonl
from .pipeline.nodes import train_from_labeled, infer as infer_nodes, evaluate as eval_nodes
from .models.persistence import save_model, load_model
from .models.calibration import ProbCalibrator
from .apps.crawler.cli import add_crawler_cli, handle_crawler
from .apps.embedder.cli import add_embedder_cli, handle_embedder
from .apps.trainer.cli import add_trainer_cli, handle_trainer
from .apps.infer.cli import add_infer_cli, handle_infer
from .apps.evaluate.cli import add_evaluate_cli, handle_evaluate

try:
    from .scripts.excel_to_training_csv import excel_to_training_csv
except Exception:
    excel_to_training_csv = None

log = get_logger("verticalizer")

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

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

def cmd_run_pipeline(geo: str, excel_path: str, labeled_csv: str, groundtruth_json: str,
                     model_path: str, calib_path: str, output_path: str, report_path: str, topk: int = 26):
    if excel_to_training_csv is None:
        raise RuntimeError("Excel converter not found.")
    log.info("[PIPE] Step 1: Converting Excel to CSV/JSON...")
    excel_to_training_csv(excel_path, geo, labeled_csv, groundtruth_json)
    log.info("[PIPE] Step 2: Training...")
    df = read_table(labeled_csv)
    bundle = train_from_labeled(df)
    _ensure_dir(model_path); _ensure_dir(calib_path); _ensure_dir(report_path)
    save_model(bundle["model"], model_path)
    bundle["cal"].save(calib_path)
    metrics = eval_nodes(bundle["model"], bundle["cal"], bundle["classes"], df)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("[PIPE] Step 3: Inference...")
    from .utils.taxonomy import load_taxonomy
    id2label, _ = load_taxonomy()
    classes = list(id2label.keys())
    out = infer_nodes(bundle["model"], bundle["cal"], classes, df, topk=topk)
    _ensure_dir(output_path)
    write_jsonl(output_path, out)
    log.info("[PIPE] Step 4: Compare predictions...")
    qa_report_path = os.path.splitext(report_path) + ".compare.json"
    cmp_report = compare_predictions(output_path, groundtruth_json, qa_report_path)
    log.info(f"[PIPE] Compare summary: {cmp_report['summary']}")

def app():
    parser = argparse.ArgumentParser(prog="verticalizer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # New modular commands
    add_crawler_cli(sub)
    add_embedder_cli(sub)
    add_trainer_cli(sub)
    add_infer_cli(sub)
    add_evaluate_cli(sub)

    # Legacy integrated pipeline
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

    if args.cmd == "run-pipeline":
        cmd_run_pipeline(args.geo, args.excel_path, args.labeled_csv, args.groundtruth_json,
                         args.model_path, args.calib_path, args.output_path, args.report_path, args.topk)
    elif args.cmd == "crawl":
        handle_crawler(args)
    elif args.cmd == "embed":
        handle_embedder(args)
    elif args.cmd == "train":
        handle_trainer(args)
    elif args.cmd == "infer":
        handle_infer(args)
    elif args.cmd == "eval":
        handle_evaluate(args)
    else:
        parser.error("Unknown command")

if __name__ == "__main__":
    app()
