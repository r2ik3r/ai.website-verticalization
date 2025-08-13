def add_embedder_cli(subparsers):
    p = subparsers.add_parser("embed", help="Embed latest crawled text and persist vectors")
    p.add_argument("--model", default="models/text-embedding-004")
    p.add_argument("--in", dest="in_path", required=True, help="CSV with 'website' column")
    p.add_argument("--no-s3", action="store_true", help="Do not store vector bytes to S3 (metadata only)")

def handle_embedder(args):
    import pandas as pd
    from .service import embed_sites
    df = pd.read_csv(args.in_path)
    sites = [str(x).strip().lower() for x in df["website"].tolist() if str(x).strip()]
    embed_sites(sites, model_name=args.model, store_to_s3=not args.no_s3)
