# src/verticalizer/apps/trainer/cli.py
def add_trainer_cli(subparsers):
    p = subparsers.add_parser("train", help="Train a model from labeled CSV")
    p.add_argument("--geo", required=True)
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--out-base", default="models")

def handle_trainer(args):
    from .service import train_from_csv
    r = train_from_csv(args.in_path, args.geo, args.version, args.out_base)
    print(json_dump(r))

def json_dump(obj):
    import json
    return json.dumps(obj, indent=2)
