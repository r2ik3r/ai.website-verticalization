import argparse
import os
import json
import pandas as pd
from .utils.logging import get_logger
from .utils.seed import seed_all
from .pipeline.io import read_table
from .pipeline.nodes import train_from_labeled, infer as infer_nodes, evaluate as eval_nodes
from .pipeline.self_train import self_training_loop
from .models.persistence import save_model, load_model
from .models.calibration import ProbCalibrator
from .pipeline.io import write_jsonl

log = get_logger("verticalizer")

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

    args = parser.parse_args()
    seed_all(42)

    if args.cmd == "train":
        df = read_table(args.in_path)
        bundle = train_from_labeled(df)
        save_model(bundle["model"], args.model_out)
        bundle["cal"].save(args.calib_out)
        # quick validation on same set (for reporting)
        metrics = eval_nodes(bundle["model"], bundle["cal"], list(bundle["classes"]) if "classes" in bundle else list(bundle["cal"].cals.keys()), df)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log.info("Training completed")

    elif args.cmd == "infer":
        df = read_table(args.in_path)
        model = load_model(args.model)
        cal = ProbCalibrator.load(args.calib) if args.calib and os.path.exists(args.calib) else ProbCalibrator()
        from .utils.taxonomy import load_taxonomy
        id2label, _ = load_taxonomy()
        classes = list(id2label.keys())
        out = infer_nodes(model, cal, classes, df, topk=args.topk)
        write_jsonl(args.out, out)
        log.info("Inference completed")

    elif args.cmd == "validate":
        df = read_table(args.in_path)
        model = load_model(args.model)
        cal = ProbCalibrator.load(args.calib) if args.calib and os.path.exists(args.calib) else ProbCalibrator()
        from .utils.taxonomy import load_taxonomy
        id2label, _ = load_taxonomy()
        classes = list(id2label.keys())
        metrics = eval_nodes(model, cal, classes, df)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log.info("Validation report written")

    elif args.cmd == "self-train":
        import pandas as pd
        seed_df = read_table(args.seed)
        unlabeled_df = read_table(args.unlabeled)
        model, cal, classes = self_training_loop(seed_df, unlabeled_df, iterations=args.iterations)
        save_model(model, args.model_out)
        cal.save(args.calib_out)
        # Report basic metrics if seed has labels
        try:
            metrics = eval_nodes(model, cal, classes, seed_df)
        except Exception:
            metrics = {"note": "No labels available for seed evaluation"}
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log.info("Self-training completed")
