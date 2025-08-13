import requests
from .robots import robots_allowed, delay, DEFAULT_UA, TIMEOUT
from .parse import extract_readable_text

def normalize_url(website: str) -> str:
    if website.startswith("http://") or website.startswith("https://"):
        return website
    return f"https://{website}"

def fetch_text(website: str) -> str:
    url = normalize_url(website)
    if not robots_allowed(url):
        return ""
    headers = {"User-Agent": DEFAULT_UA, "Accept": "text/html,application/xhtml+xml"}
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers=headers)
        if 200 <= resp.status_code < 300 and "text/html" in resp.headers.get("Content-Type", ""):
            delay()
            return extract_readable_text(resp.text)[:200000]
        return ""
    except Exception:
        return ""
