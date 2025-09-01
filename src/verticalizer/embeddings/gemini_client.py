# src/verticalizer/embeddings/gemini_client.py
import os
import time
import hashlib
import logging
from typing import List, Tuple

try:
    from tqdm.rich import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from google import genai
from google.genai import types
from google.api_core import retry as gretry

from .cache import get_cached, set_cached

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_EMB_MODEL", "models/text-embedding-004")
EMBED_DIM = int(os.getenv("GEMINI_EMB_DIM", "768"))
TASK_TYPE = os.getenv("GEMINI_TASK_TYPE", "classification")
DRYRUN = bool(int(os.getenv("GEMINI_EMB_DRYRUN", "0")))
MAX_CALLS = int(os.getenv("GEMINI_EMB_MAX_CALLS", "0"))
RATE_LIMIT_QPS = float(os.getenv("GEMINI_EMB_RATELIMIT", "0"))

if not GEMINI_API_KEY and not DRYRUN:
    raise RuntimeError("GEMINI_API_KEY not set in env")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

_last_call_ts = 0.0

def _rate_limit():
    global _last_call_ts
    if RATE_LIMIT_QPS > 0:
        min_interval = 1.0 / RATE_LIMIT_QPS
        now = time.time()
        wait = min_interval - (now - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.time()

def _is_retriable(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    return code in (429, 503)

class GeminiEmbedder:
    def __init__(self, model: str = MODEL, task_type: str = TASK_TYPE, embeddim: int = EMBED_DIM):
        self.model = model
        self.task_type = task_type
        self.embeddim = embeddim
        self.calls = 0
        self.retry = gretry.Retry(predicate=_is_retriable, deadline=300.0)

    def embed_text(self, text: str) -> List[float]:
        if not text or not str(text).strip():
            return [0.0] * self.embeddim

        norm = str(text).strip()
        cached = get_cached(norm, self.model)
        if cached is not None:
            return cached

        if DRYRUN:
            vec = [0.0] * self.embeddim
            set_cached(norm, self.model, vec)
            return vec

        if MAX_CALLS and self.calls >= MAX_CALLS:
            vec = [0.0] * self.embeddim
            set_cached(norm, self.model, vec)
            return vec

        _rate_limit()
        try:
            resp = client.models.embed_content(
                model=self.model,
                contents=norm[:100000],
                config=types.EmbedContentConfig(task_type=self.task_type),
            )
        except Exception as e:
            snip = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]
            logger.error("GeminiEmbedder: failed to embed text %s: %s", snip, e)
            raise

        if not getattr(resp, "embeddings", None) or not hasattr(resp.embeddings, "values"):
            raise ValueError(f"GeminiEmbedder: no embedding values for model {self.model}")

        vec = list(resp.embeddings.values)
        set_cached(norm, self.model, vec)
        self.calls += 1
        return vec

    def embed_texts_dedup(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        # normalize and de-duplicate
        order: List[Tuple[int, str]] = []
        unique_map = {}
        for i, t in enumerate(texts):
            nt = (t if isinstance(t, str) else "").strip()
            order.append((i, nt))
            if nt not in unique_map:
                unique_map[nt] = None

        # cache hits
        cached_hits = 0
        for nt in list(unique_map.keys()):
            cv = get_cached(nt, self.model)
            if cv is not None:
                unique_map[nt] = cv
                cached_hits += 1

        # misses
        misskeys = [nt for nt, v in unique_map.items() if v is None]
        iterator = tqdm(misskeys, desc="Embedding unique texts", unit="doc") if show_progress else misskeys
        for nt in iterator:
            unique_map[nt] = self.embed_text(nt)

        total_unique = len(unique_map)
        if total_unique:
            hitrate = cached_hits / total_unique
            logger.info("GeminiEmbedder: cache hits %d/%d (%.1f%%) on unique texts", cached_hits, total_unique, 100*hitrate)

        # map back to original order
        out = []
        for _, nt in order:
            vec = unique_map.get(nt) or ([0.0] * self.embeddim)
            out.append(vec)
        return out

    def embed_dataframe_column(self, df, column: str, new_column: str = "embedding", show_progress: bool = True):
        from pandas import Series
        series = Series(df[column].fillna("").astype(str))
        embeddings = self.embed_texts_dedup(series.tolist(), show_progress=show_progress)
        df[new_column] = embeddings
        return df