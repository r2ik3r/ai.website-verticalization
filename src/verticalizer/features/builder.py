from typing import List, Dict
import pandas as pd
from ..crawl.fetcher import fetch_text
from ..embeddings.gemini_client import GeminiEmbedder

def crawl_and_embed(df: pd.DataFrame, text_col: str | None = "content_text") -> pd.DataFrame:
    emb = GeminiEmbedder()
    texts: List[str] = []
    for _, row in df.iterrows():
        txt = row.get(text_col) if text_col and text_col in df.columns and isinstance(row.get(text_col), str) else ""
        if not txt:
            txt = fetch_text(row["website"])
        texts.append(txt)
    df = df.copy()
    df["content_text"] = texts
    df["embedding"] = [emb.embed_text(t) for t in texts]
    return df
w