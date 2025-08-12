import os
import requests
from tenacity import retry, wait_exponential, stop_after_attempt
from .cache import get_cached, set_cached

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_EMB_MODEL", "text-embedding-004")
BASE = os.environ.get("GEMINI_EMB_BASE", "https://generativelanguage.googleapis.com/v1beta/models")

class GeminiEmbedder:
    def __init__(self, api_key: str | None = None, model: str = MODEL):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY missing")
        self.model = model

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5))
    def embed_text(self, text: str) -> list[float]:
        if not text:
            return [0.0] * 768
        cached = get_cached(text, self.model)
        if cached is not None:
            return cached
        url = f"{BASE}/{self.model}:embedText?key={self.api_key}"
        r = requests.post(url, json={"text": text[:100000]}, timeout=30)
        r.raise_for_status()
        vec = r.json()["embedding"]["value"]
        set_cached(text, self.model, vec)
        return vec
