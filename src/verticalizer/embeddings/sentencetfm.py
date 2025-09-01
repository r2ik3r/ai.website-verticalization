# src/verticalizer/embeddings/sentencetfm.py

import hashlib
import os
from typing import List, Tuple, Dict

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

from .cache import get_cached as _get_cached, set_cached as _set_cached  # reuse existing cache interface if available
# Fallback to gemini cache API names in repo
try:
    from .cache import getcached as getcached_vec, setcached as setcached_vec
except Exception:
    def getcached_vec(text: str, model: str):
        return _get_cached(text, model) if '_get_cached' in globals() else None
    def setcached_vec(text: str, model: str, vec):
        return _set_cached(text, model, vec) if '_set_cached' in globals() else None

DEFAULT_MODEL = os.getenv("SENTENCE_TFM_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_DIM = int(os.getenv("SENTENCE_TFM_DIM", "384"))
DRYRUN = bool(int(os.getenv("SENTENCE_TFM_DRYRUN", "0")))

class SentenceTfmEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, embeddim: int = DEFAULT_DIM):
        self.model_name = model_name
        self.embeddim = embeddim
        self.model = None if DRYRUN or SentenceTransformer is None else SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        if not text or not str(text).strip():
            return [0.0] * self.embeddim
        norm = str(text).strip()
        cached = getcached_vec(norm, self.model_name)
        if cached is not None:
            return cached
        if DRYRUN or self.model is None:
            vec = [0.0] * self.embeddim
            setcached_vec(norm, self.model_name, vec)
            return vec
        vec = self.model.encode([norm], normalize_embeddings=False).tolist()
        setcached_vec(norm, self.model_name, vec)
        return vec

    def embed_texts_dedup(self, texts: List[str]) -> List[List[float]]:
        order: List[Tuple[int, str]] = []
        uniq: Dict[str, List[float]] = {}
        for i, t in enumerate(texts):
            nt = str(t or "").strip()
            order.append((i, nt))
            uniq.setdefault(nt, None)
        # Fill cache hits
        for k in list(uniq.keys()):
            cv = getcached_vec(k, self.model_name)
            if cv is not None:
                uniq[k] = cv
        # Compute misses
        misses = [k for k, v in uniq.items() if v is None]
        if not DRYRUN and self.model is not None and misses:
            embs = self.model.encode(misses, normalize_embeddings=False).tolist()
            for k, v in zip(misses, embs):
                uniq[k] = v
                setcached_vec(k, self.model_name, v)
        # Fill remaining with zeros
        for k in list(uniq.keys()):
            if uniq[k] is None:
                uniq[k] = [0.0] * self.embeddim
        # Restore order
        out: List[List[float]] = []
        for _, nt in order:
            out.append(uniq.get(nt, [0.0] * self.embeddim))
        return out