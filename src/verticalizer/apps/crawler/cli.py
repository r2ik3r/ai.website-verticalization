# src/verticalizer/apps/crawler/cli.py (replace file)

def addcrawlerclisubparsers(p):
    sub = p.add_parser("crawl", help="Crawl websites and persist content")
    sub.add_argument("--in", dest="inpath", required=False, help="CSV with website column")
    sub.add_argument("--sites", nargs="*", help="List of sites e.g., cnn.com webmd.com")
    sub.add_argument("--urls-csv", dest="urlcsv", required=False, help="CSV with columns: website,url")
    sub.add_argument("--store-html", action="store_true", help="Store raw HTML to S3/object storage")

def handlecrawlerargs(args):
    import pandas as pd
    from .service import crawl_sites, crawl_site_urls
    sites = []
    if args.inpath:
        df = pd.read_csv(args.inpath)
        if "website" in df.columns:
            sites = [str(x).strip().lower() for x in df["website"].tolist() if str(x).strip()]
    if args.sites:
        sites.extend([s.strip().lower() for s in args.sites if s.strip()])
    if args.urlcsv:
        dfu = pd.read_csv(args.urlcsv)
        if not {"website", "url"}.issubset(dfu.columns):
            raise SystemExit("urls-csv must contain columns: website,url")
        crawl_site_urls(dfu, store_html=args.store_html)
        return
    if not sites:
        raise SystemExit("No sites provided. Use --in, --sites, or --urls-csv")
    crawl_sites(sites, source="manual", store_html=args.store_html)