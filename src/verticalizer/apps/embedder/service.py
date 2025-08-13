import hashlib
import logging
from typing import List
from ...storage.repositories import create_tables_if_missing, latest_text_for_site_batch, record_embedding
from ...storage.s3 import put_bytes
from ...embeddings.gemini_client import GeminiEmbedder

logger = logging.getLogger(__name__)

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def embed_sites(sites: List[str], model_name: str, store_to_s3: bool = True):
    create_tables_if_missing()
    texts = latest_text_for_site_batch(sites)
    embedder = GeminiEmbedder(model=model_name)
    pairs = [(s, texts.get(s) or "") for s in sites]
    vectors = embedder.embed_texts_dedup([t for _, t in pairs], show_progress=True)
    import numpy as np
    for (site, text), vec in zip(pairs, vectors):
        sha = _sha256(text)
        key = f"embeddings/{site}/{model_name}/{sha}.npy"
        arr = np.array(vec, dtype="float32")
        vector_ref = ""
        if store_to_s3:
            put_bytes(key, arr.tobytes(), "application/octet-stream")
            vector_ref = key
        record_embedding(site=site, model_name=model_name, dim=len(vec), sha_text=sha, vector_ref=vector_ref, vector_len=len(vec))
