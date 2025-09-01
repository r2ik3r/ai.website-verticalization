# src/verticalizer/apps/crawler/service.py

import hashlib
import logging
from typing import List
import pandas as pd

from ...crawl.fetcher import fetch_text
from ...storage.repositories import create_tables_if_missing, record_crawl, upsert_site

logger = logging.getLogger(__name__)

def _sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()

def crawl_sites(sites: List[str], source: str = "batch", store_html: bool = False):
    from ...storage.s3 import putbytes
    create_tables_if_missing()
    for site in sites:
        try:
            upsert_site(site)
            text, html, status, lang = fetch_text(site, return_html=True)  # returns tuple
            excerpt = (text or "")[:4000]
            content_hash = _sha256(text or html or "")
            htmlkey = None
            if store_html and html:
                htmlkey = f"raw-html/{site}/{content_hash}.html"
                putbytes(htmlkey, (html or "").encode("utf-8", errors="ignore"), "text/html; charset=utf-8")
            record_crawl(
                site=site,
                url=f"https://{site}",
                httpstatus=int(status or 0),
                contenthash=content_hash,
                textexcerpt=excerpt,
                textfullref=htmlkey or "",
                lang=lang or "en",
                source=source,
                status="OK" if status and 200 <= int(status) < 400 else "ERROR",
            )
        except Exception as e:
            logger.exception("crawler Failed %s: %s", site, e)
            record_crawl(
                site=site,
                url=f"https://{site}",
                httpstatus=0,
                contenthash="",
                textexcerpt="",
                textfullref="",
                lang="",
                source=source,
                status="ERROR",
            )

def crawl_site_urls(df_urls: pd.DataFrame, store_html: bool = False):
    """
    Crawl per (website,url) row for multi-URL site aggregation.
    """
    from ...storage.s3 import putbytes
    create_tables_if_missing()
    for _, row in df_urls.iterrows():
        site = str(row["website"]).strip().lower()
        url = str(row["url"]).strip()
        if not site or not url:
            continue
        try:
            upsert_site(site)
            text, html, status, lang = fetch_text(url, return_html=True)
            excerpt = (text or "")[:4000]
            content_hash = _sha256(text or html or "")
            htmlkey = None
            if store_html and html:
                htmlkey = f"raw-html/{site}/{content_hash}.html"
                putbytes(htmlkey, (html or "").encode("utf-8", errors="ignore"), "text/html; charset=utf-8")
            record_crawl(
                site=site,
                url=url,
                httpstatus=int(status or 0),
                contenthash=content_hash,
                textexcerpt=excerpt,
                textfullref=htmlkey or "",
                lang=lang or "en",
                source="urls_csv",
                status="OK" if status and 200 <= int(status) < 400 else "ERROR",
            )
        except Exception as e:
            logger.exception("crawler URL Failed %s %s: %s", site, url, e)
            record_crawl(
                site=site,
                url=url,
                httpstatus=0,
                contenthash="",
                textexcerpt="",
                textfullref="",
                lang="",
                source="urls_csv",
                status="ERROR",
            )