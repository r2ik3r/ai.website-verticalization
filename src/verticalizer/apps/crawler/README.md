# Crawler Module

Purpose
- Fetch site content (text + optional HTML), persist to Postgres (+ S3).

Input
- CSV with website column or --sites list.

Output
- tables: sites, crawls; optional S3 raw_html objects.

Commands
- poetry run verticalizer crawl --in data/sites.csv --store-html
- poetry run verticalizer crawl --sites cnn.com webmd.com

Notes
- Respects fetcher throttling and robots; stores content_hash for incremental updates.
- Pairs with embedder module for decoupled embeddings.
