# src/verticalizer/apps/trainer/cli.py

def addtrainerclisubparsers(p):
    sub = p.add_parser("train", help="Train a model from labeled CSV")
    sub.add_argument("--geo", required=True)
    sub.add_argument("--in", dest="inpath", required=True)
    sub.add_argument("--version", required=True)
    sub.add_argument("--out-base", dest="outbase", default="models")
    # Hyperparameters
    sub.add_argument("--epochs", type=int, default=15)
    sub.add_argument("--batch-size", type=int, default=64)
    sub.add_argument("--hidden", type=int, default=512)
    sub.add_argument("--dropout", type=float, default=0.3)
    sub.add_argument("--labels-loss", default="focal", choices=["bce","focal"])
    sub.add_argument("--gamma", type=float, default=2.0)
    sub.add_argument("--val-split", type=float, default=0.2)
    sub.add_argument("--early-stop", action="store_true")
    sub.add_argument("--iab-version", default="v3")

def handletrainerargs(args):
    from .service import train_from_csv
    cfg = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        dropout=args.dropout,
        labels_loss=args.labels_loss,
        gamma=args.gamma,
        val_split=args.val_split,
        early_stop=args.early_stop,
        iab_version=args.iab_version,
    )
    r = train_from_csv(args.inpath, args.geo, args.version, args.outbase, cfg)
    print(json_dump(r))

def json_dump(obj):
    import json
    return json.dumps(obj, indent=2)