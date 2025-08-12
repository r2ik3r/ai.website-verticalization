import time
from urllib.parse import urljoin, urlparse
import urllib.robotparser as robotparser
import requests
import os

DEFAULT_UA = os.environ.get("HTTP_USER_AGENT", "IABVerticalizerBot/1.0")
TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "20"))

def robots_allowed(url: str, user_agent: str = DEFAULT_UA) -> bool:
    parsed = urlparse(url if url.startswith("http") else f"https://{url}")
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = robotparser.RobotFileParser()
    try:
        resp = requests.get(urljoin(base, "/robots.txt"), timeout=TIMEOUT, headers={"User-Agent": user_agent})
        if resp.status_code >= 400:
            return True
        rp.parse(resp.text.splitlines())
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True

def delay():
    time.sleep(0.7)
