# src/verticalizer/apps/infer/cli.py

def addinferclisubparsers(p):
    sub = p.add_parser("infer", help="Run inference on CSV")
    sub.add_argument("--in", dest="inpath", required=True)
    sub.add_argument("--model", dest="model", required=False, help="Single model path")
    sub.add_argument("--calib", dest="calib", required=False, default=None)
    sub.add_argument("--models", nargs="*", default=None, help="Multiple model paths for ensembling")
    sub.add_argument("--calibs", nargs="*", default=None, help="Multiple calibrator paths for ensembling")
    sub.add_argument("--out", required=True)
    sub.add_argument("--topk", type=int, default=10)
    sub.add_argument("--iab-version", default="v3")
    sub.add_argument("--hierarchy-consistent", action="store_true")
    sub.add_argument("--group-col", default=None)
    sub.add_argument("--url-col", default=None)
    sub.add_argument("--page-agg", default="mean", choices=["mean", "softmax_mean"])
    sub.add_argument("--ensemble-method", default="mean", choices=["mean", "softmax_mean"])

def handleinferargs(args):
    from .service import infer_from_csv
    infer_from_csv(
        args.inpath,
        args.model,
        args.calib,
        args.out,
        args.topk,
        args.models,
        args.calibs,
        args.iab_version,
        args.hierarchy_consistent,
        args.group_col,
        args.url_col,
        args.page_agg,
        args.ensemble_method,
    )