# src/verticalizer/crawl/fetcher.py
import logging
import requests
from .robots import robots_allowed, delay, DEFAULT_UA, TIMEOUT
from .parse import extract_readable_text

logger = logging.getLogger(__name__)

def normalize_url(website: str) -> str:
    website = (website or "").strip()
    if website.startswith("http://") or website.startswith("https://"):
        return website
    return f"https://{website}"

def fetch_text(website: str, return_html: bool = False):
    """
    Returns (text, html, status, lang) with robots.txt respect and defensive guards.
    - text: readable extracted excerpt
    - html: raw HTML if requested (else None)
    - status: HTTP status or 0 on failure
    - lang: best-effort language code (None if unknown)
    """
    url = normalize_url(website)
    if not robots_allowed(url):
        logger.warning("ROBOTS disallow: %s", url)
        return "", None, 403, None

    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers=headers)
        status = resp.status_code
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if 200 <= status < 300 and ("text/html" in ctype or "application/xhtml+xml" in ctype):
            delay()
            html = resp.text or ""
            text = extract_readable_text(html) or ""
            # very coarse lang best-effort; could be improved with fasttext or CLD3 offline
            lang = (resp.headers.get("Content-Language") or "").split(",").strip() or None
            return text, (html if return_html else None), status, (lang or "en")
        else:
            logger.warning("Non-HTML or bad status for %s: %s %s", url, status, ctype)
            return "", (resp.text if return_html else None), status, None
    except Exception as e:
        logger.exception("fetch_text failed for %s: %s", url, e)
        return "", None, 0, None