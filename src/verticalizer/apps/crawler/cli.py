def add_crawler_cli(subparsers):
    p = subparsers.add_parser("crawl", help="Crawl websites and persist content")
    p.add_argument("--in", dest="in_path", required=False, help="CSV with 'website' column")
    p.add_argument("--sites", nargs="*", help="List of sites (e.g., cnn.com webmd.com)")
    p.add_argument("--store-html", action="store_true", help="Store raw HTML to S3/object storage")

def handle_crawler(args):
    import pandas as pd
    from .service import crawl_sites
    sites = []
    if args.in_path:
        df = pd.read_csv(args.in_path)
        sites = [str(x).strip().lower() for x in df["website"].tolist() if str(x).strip()]
    if args.sites:
        sites.extend([s.strip().lower() for s in args.sites if s.strip()])
    if not sites:
        raise SystemExit("No sites provided. Use --in or --sites")
    crawl_sites(sites, source="manual", store_html=args.store_html)
