from bs4 import BeautifulSoup
from readability import Document

def extract_readable_text(html: str) -> str:
    try:
        doc = Document(html)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "lxml")
    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    texts = []
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    if title:
        texts.append(title)
    metas = soup.find_all("meta")
    for m in metas:
        if m.get("name") in {"description", "og:description"} and m.get("content"):
            texts.append(m["content"])
    h1 = soup.find_all("h1")
    for h in h1[:5]:
        if h.get_text(strip=True):
            texts.append(h.get_text(strip=True))
    body = soup.get_text(separator=" ", strip=True)
    if body:
        texts.append(body)
    text = " ".join(t for t in texts if t)
    return " ".join(text.split())
