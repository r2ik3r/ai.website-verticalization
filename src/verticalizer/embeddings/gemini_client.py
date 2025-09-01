# src/verticalizer/embeddings/gemini_client.py
import os
import hashlib
import logging
import time
from typing import List

try:
    from tqdm.rich import tqdm
except ImportError:
    from tqdm import tqdm

from google import genai
from google.genai import types
from google.api_core import retry as g_retry

from .cache import get_cached, set_cached

logger = logging.getLogger(__name__)

# === Environment config ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_EMB_MODEL", "models/text-embedding-004")
EMBED_DIM = int(os.getenv("GEMINI_EMB_DIM", "768"))
TASK_TYPE = os.getenv("GEMINI_TASK_TYPE", "classification")

# Cost-safety controls
DRYRUN = os.getenv("GEMINI_EMB_DRYRUN", "0") == "1"            # If 1, skip API calls, fill zero vectors
MAX_CALLS = int(os.getenv("GEMINI_EMB_MAX_CALLS", "0"))        # Max API calls per run, 0 = unlimited
RATE_LIMIT_QPS = float(os.getenv("GEMINI_EMB_RATE_LIMIT", "0")) # Limit QPS, 0 = no limit

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in env")

# Global client
client = genai.Client(api_key=GEMINI_API_KEY)

# Retry predicate
def _is_retriable(exc: Exception) -> bool:
    return hasattr(exc, "code") and exc.code in {429, 503}

# Simple rate limiter
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


class GeminiEmbedder:
    def __init__(self, model: str = MODEL, task_type: str = TASK_TYPE, embed_dim: int = EMBED_DIM):
        self.model = model
        self.task_type = task_type
        self.embed_dim = embed_dim
        self._calls = 0

    @g_retry.Retry(predicate=_is_retriable, timeout=300.0)
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text with caching and cost-safety controls."""
        if not text or not text.strip():
            return [0.0] * self.embed_dim

        norm_text = text.strip()
        cached = get_cached(norm_text, self.model)
        if cached is not None:
            return cached

        if DRYRUN:
            return [0.0] * self.embed_dim
        if MAX_CALLS and self._calls >= MAX_CALLS:
            return [0.0] * self.embed_dim

        _rate_limit()

        try:
            resp = client.models.embed_content(
                model=self.model,
                contents=norm_text[:100_000],
                config=types.EmbedContentConfig(task_type=self.task_type),
            )
        except Exception as e:
            snippet = hashlib.sha1(norm_text.encode("utf-8")).hexdigest()[:8]
            logger.error(f"[GeminiEmbedder] Failed to embed text hash={snippet}: {e}")
            raise

        if not resp.embeddings or not hasattr(resp.embeddings[0], "values"):
            raise ValueError(f"[GeminiEmbedder] No embedding values returned for model {self.model}")

        vec = list(resp.embeddings[0].values)
        set_cached(norm_text, self.model, vec)
        self._calls += 1
        return vec

    def embed_texts_dedup(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Embed multiple texts with deduplication and progress display."""
        # Normalize & deduplicate
        norm_texts = [(i, (t.strip() if isinstance(t, str) else "")) for i, t in enumerate(texts)]
        unique_map = {}
        order = []
        for idx, nt in norm_texts:
            order.append((idx, nt))
            if nt not in unique_map:
                unique_map[nt] = None

        # Cache hits
        cached_hits = 0
        for nt in list(unique_map.keys()):
            cv = get_cached(nt, self.model)
            if cv is not None:
                unique_map[nt] = cv
                cached_hits += 1

        # Misses
        miss_keys = [nt for nt, vec in unique_map.items() if vec is None and nt]
        iterator = tqdm(miss_keys, desc="Embedding unique texts", unit="doc") if show_progress else miss_keys
        for nt in iterator:
            vec = self.embed_text(nt)
            unique_map[nt] = vec

        total_unique = len(unique_map)
        if total_unique:
            hit_rate = cached_hits / total_unique
            logger.info(f"[GeminiEmbedder] Cache hits {cached_hits}/{total_unique} ({hit_rate:.1%}) on unique texts")

        # Map back to original order
        results = []
        for _, nt in order:
            vec = unique_map.get(nt) or [0.0] * self.embed_dim
            results.append(vec)
        return results

    def embed_dataframe_column(self, df, column: str, new_column: str = "embedding", show_progress: bool = True):
        """Embed a Pandas DataFrame column with dedup and tqdm progress bar."""
        from pandas import Series
        series: Series = df[column].fillna("").astype(str)
        embeddings = self.embed_texts_dedup(series.tolist(), show_progress=show_progress)
        df[new_column] = embeddings
        return df
