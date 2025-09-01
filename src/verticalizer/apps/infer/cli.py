# src/verticalizer/apps/infer/cli.py
def add_infer_cli(subparsers):
    p = subparsers.add_parser("infer", help="Run inference on CSV of websites")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--calib", required=False, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--topk", type=int, default=10)

def handle_infer(args):
    from .service import infer_from_csv
    infer_from_csv(args.in_path, args.model, args.calib, args.out, args.topk)
