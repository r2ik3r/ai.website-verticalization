# src/verticalizer/pipeline/common.py
import logging
import numpy as np
import pandas as pd
from typing import List

from ..apps.crawler.service import crawl_sites
from ..apps.embedder.service import embed_sites
from ..storage.repositories import latest_text_for_site_batch
from ..embeddings.cache import get_cached
from ..embeddings.gemini_client import GeminiEmbedder

logger = logging.getLogger(__name__)

def prepare_embeddings_for_df(
        df: pd.DataFrame,
        model_name: str = "models/text-embedding-004",
        store_to_s3: bool = False
) -> np.ndarray:
    """
    Ensure crawls + embeddings exist for the given DataFrame and return embedding matrix X.

    Steps:
    1. Crawl sites (if needed) and persist to DB/Object Storage
    2. Embed crawled text using GeminiEmbedder (with caching)
    3. Load embeddings from cache/DB into a NumPy float32 array
    """
    sites: List[str] = df["website"].dropna().astype(str).str.strip().tolist()

    # Crawl
    logger.info(f"[COMMON] Crawling {len(sites)} sites")
    crawl_sites(sites)

    # Embed
    logger.info(f"[COMMON] Embedding {len(sites)} sites using model '{model_name}'")
    embed_sites(sites, model_name=model_name, store_to_s3=store_to_s3)

    # Load from DB/cache
    texts_map = latest_text_for_site_batch(sites)
    embedder = GeminiEmbedder(model=model_name)

    vectors = []
    for site in sites:
        text = texts_map.get(site) or ""
        vec = get_cached(text.strip(), embedder.model) or [0.0] * embedder.embed_dim
        vectors.append(vec)

    X = np.array(vectors, dtype=np.float32)
    return X
