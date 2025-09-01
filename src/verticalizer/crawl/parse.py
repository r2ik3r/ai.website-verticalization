# src/verticalizer/crawl/parse.py
from bs4 import BeautifulSoup
from readability import Document

MAX_CHARS = 200000  # bound for downstream embedding API

def extract_readable_text(html: str) -> str:
    html = html or ""
    try:
        doc = Document(html)
        summary_html = doc.summary() or html
        soup = BeautifulSoup(summary_html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        try:
            tag.decompose()
        except Exception:
            tag.extract()

    texts = []
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    if title:
        texts.append(title)

    for m in soup.find_all("meta"):
        name = (m.get("name") or m.get("property") or "").lower()
        if name in ("description", "og:description"):
            content = (m.get("content") or "").strip()
            if content:
                texts.append(content)

    for h in soup.find_all("h1")[:5]:
        t = h.get_text(strip=True)
        if t:
            texts.append(t)

    body = soup.get_text(separator=" ", strip=True)
    if body:
        texts.append(body)

    text = "\n".join(t for t in texts if t)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return text