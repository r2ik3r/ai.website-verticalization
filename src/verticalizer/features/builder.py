# src/verticalizer/features/builder.py
from typing import List
import pandas as pd
import numpy as np

from ..crawl.fetcher import fetch_text
from ..embeddings.gemini_client import GeminiEmbedder


def _ensure_content_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a content_text column.
    If missing or empty, crawl the website homepage to extract readable text.
    """
    df = df.copy()
    if "content_text" not in df.columns:
        df["content_text"] = ""

    texts: List[str] = []
    for _, row in df.iterrows():
        txt = row.get("content_text")
        if isinstance(txt, str) and txt.strip():
            texts.append(txt)
            continue
        site = str(row.get("website", "")).strip()
        if not site:
            texts.append("")
            continue
        try:
            crawled = fetch_text(site)
            texts.append(crawled or "")
        except Exception:
            texts.append("")
    df["content_text"] = texts
    return df


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using Gemini with local caching.
    """
    embedder = GeminiEmbedder()
    vectors: List[List[float]] = []
    for t in texts:
        try:
            vec = embedder.embed_text(t)
            vectors.append(vec)
        except Exception:
            # If embedding fails, use a zero vector of common dimension (e.g., 768).
            # This is a safe fallback; downstream model expects fixed-length vectors.
            vectors.append([0.0] * 768)
    return vectors


def crawl_and_embed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Public API used by training/inference:
    - Ensure content_text is present (crawl if missing).
    - Embed content_text into a fixed-length embedding.
    - Return df with 'content_text' and 'embedding' columns populated.
    """
    df = _ensure_content_text(df)
    texts = df["content_text"].fillna("").astype(str).tolist()
    vectors = _embed_texts(texts)

    # Validate consistent embedding shapes
    # If any vector is not the same size, pad/truncate to the dominant length.
    lengths = [len(v) for v in vectors]
    if not lengths:
        emb_dim = 768
        vectors = [[0.0] * emb_dim]
    else:
        # Choose the most common length as canonical
        emb_dim = max(set(lengths), key=lengths.count)
        fixed = []
        for v in vectors:
            if len(v) == emb_dim:
                fixed.append(v)
            elif len(v) > emb_dim:
                fixed.append(v[:emb_dim])
            else:
                fixed.append(v + [0.0] * (emb_dim - len(v)))
        vectors = fixed

    df["embedding"] = vectors
    return df
