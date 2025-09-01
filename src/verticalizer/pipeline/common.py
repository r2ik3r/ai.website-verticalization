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

def prepare_embeddings_for_df(df: pd.DataFrame, modelname: str = "models/text-embedding-004", store_to_s3: bool = False) -> np.ndarray:
    # Ensure crawl and embeddings exist
    sites: List[str] = (
        df["website"].dropna().astype(str).str.strip().tolist()
        if "website" in df.columns else []
    )
    sites = [s for s in sites if s]

    if not sites:
        return np.zeros((0, 768), dtype=np.float32)

    logger.info("COMMON: Crawling %d sites", len(sites))
    crawl_sites(sites)

    embed_sites(sites, modelname=modelname, store_to_s3=store_to_s3)

    texts_map = latest_text_for_site_batch(sites)
    embedder = GeminiEmbedder(model=modelname)
    vectors = []
    for site in sites:
        text = texts_map.get(site) or ""
        vec = get_cached(text.strip(), embedder.model) or embedder.embed_text(text)
        vectors.append(vec)

    X = np.array(vectors, dtype=np.float32)
    return X