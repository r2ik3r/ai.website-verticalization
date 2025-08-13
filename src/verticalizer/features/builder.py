# src/verticalizer/features/builder.py

import pandas as pd
import logging
from typing import List
from ..crawl.fetcher import fetch_text
from ..embeddings.gemini_client import GeminiEmbedder

# Use tqdm for progress
try:
    from tqdm.rich import tqdm
except ImportError:
    from tqdm import tqdm

logger = logging.getLogger(__name__)

def _ensure_content_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'content_text' column exists and populated.
    Crawl missing ones with a visible progress bar.
    """
    df = df.copy()
    if "content_text" not in df.columns:
        df["content_text"] = ""

    texts: List[str] = []
    iterator = tqdm(df.iterrows(), total=len(df), desc="Crawling sites", unit="site")
    for _, row in iterator:
        txt = row.get("content_text")
        if isinstance(txt, str) and txt.strip():
            texts.append(txt)
        else:
            site = str(row.get("website", "")).strip()
            if not site:
                texts.append("")
                continue
            try:
                crawled = fetch_text(site)
                texts.append(crawled or "")
            except Exception as e:
                logger.warning(f"[crawl_and_embed] Failed to crawl {site}: {e}")
                texts.append("")
    df["content_text"] = texts
    return df


def crawl_and_embed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline step:
    1. Crawl + fill content_text (with progress bar)
    2. Embed content_text using GeminiEmbedder (with progress & cache stats)
    """
    df = _ensure_content_text(df)

    logger.info(f"[crawl_and_embed] Embedding {len(df)} documents with Gemini...")
    embedder = GeminiEmbedder()
    df = embedder.embed_dataframe_column(df, "content_text", "embedding", show_progress=True)

    # Sanity check: ensure consistent embedding lengths
    lengths = [len(v) if isinstance(v, list) else 0 for v in df["embedding"]]
    if len(set(lengths)) != 1:
        max_len = max(lengths)
        logger.warning(f"[crawl_and_embed] Inconsistent embedding sizes detected, padding to {max_len}")
        fixed = []
        for v in df["embedding"]:
            if len(v) == max_len:
                fixed.append(v)
            elif len(v) > max_len:
                fixed.append(v[:max_len])
            else:
                fixed.append(v + [0.0] * (max_len - len(v)))
        df["embedding"] = fixed

    return df
