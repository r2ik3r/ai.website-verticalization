# src/verticalizer/apps/evaluate/cli.py
def add_evaluate_cli(subparsers):
    p = subparsers.add_parser("eval", help="Compare predictions vs ground truth")
    p.add_argument("--pred", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--out", required=True)

def handle_evaluate(args):
    from .service import compare_jsonl_to_gold
    r = compare_jsonl_to_gold(args.pred, args.gold, args.out)
    print(json_dump(r))

def json_dump(obj):
    import json
    return json.dumps(obj, indent=2)
