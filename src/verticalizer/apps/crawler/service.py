# src/verticalizer/apps/crawler/service.py
import hashlib
import logging
from typing import List
from ...crawl.fetcher import fetch_text
from ...storage.repositories import create_tables_if_missing, record_crawl, upsert_site
from ...storage.s3 import put_bytes

logger = logging.getLogger(__name__)

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def crawl_sites(sites: List[str], source: str = "batch", store_html: bool = False):
    create_tables_if_missing()
    for site in sites:
        try:
            upsert_site(site)
            try:
                # preferred signature
                text, html, status, lang = fetch_text(site, return_html=True)
            except TypeError:
                # fallback if fetch_text only returns text
                text = fetch_text(site)
                html, status, lang = "", 200, "en"
            content_hash = _sha256(text or "")
            excerpt = (text or "")[:4000]
            html_key = ""
            if store_html and html:
                html_key = f"raw_html/{site}/{content_hash}.html"
                put_bytes(html_key, html.encode("utf-8"), "text/html")
            record_crawl(
                site=site,
                url=f"https://{site}",
                http_status=status or 200,
                content_hash=content_hash,
                text_excerpt=excerpt,
                text_full_ref=html_key,
                lang=lang or "en",
                source=source,
                status="OK"
            )
        except Exception as e:
            logger.exception(f"[crawler] Failed {site}: {e}")
            record_crawl(
                site=site, url=f"https://{site}", http_status=0, content_hash="", text_excerpt="", text_full_ref="", lang="", source=source, status="ERROR"
            )
