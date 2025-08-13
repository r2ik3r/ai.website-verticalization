import os
import pandas as pd
from ..embeddings.gemini_client import GeminiEmbedder

# Optional: configure your cost per 1K embeddings here (USD)
# Refer to Google API pricing: https://cloud.google.com/vertex-ai/pricing#text_embeddings
PRICE_PER_1K = float(os.getenv("EMBED_PRICE_PER_1K", "0.0001"))  # example: $0.0001 per 1K tokens

def prewarm_and_report(csv_path: str, text_column: str = "content_text"):
    print(f"📄 Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {csv_path}")

    texts = df[text_column].fillna("").astype(str).tolist()
    unique_texts = set(texts)

    print(f"📊 Found {len(df)} rows, {len(unique_texts)} unique texts")

    embedder = GeminiEmbedder()

    # Run pre-warm (dedup with tqdm)
    embedder.embed_texts_dedup(list(unique_texts), show_progress=True)

    calls_made = embedder._calls
    cost_est = calls_made * PRICE_PER_1K  # simplistic cost estimate (per 1K items)

    print("\n💰 Embedding Cost Report")
    print(f"   Unique texts embedded:  {len(unique_texts)}")
    print(f"   Cache hits:             {len(unique_texts)-calls_made}")
    print(f"   API calls made:         {calls_made}")
    print(f"   Estimated API cost:     ${cost_est:.6f} USD (at ${PRICE_PER_1K}/1K embeds)")
    print("✅ Cache is now warmed — next pipeline run will reuse embeddings and cost $0.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pre-warm the Gemini embedding cache and report cost.")
    parser.add_argument("--csv", required=True, help="Path to the labeled CSV file")
    parser.add_argument("--col", default="content_text", help="Name of text column to embed (default: content_text)")
    args = parser.parse_args()

    prewarm_and_report(args.csv, args.col)

# PYTHONPATH=./src poetry run python verticalizer/scripts/prewarm_embeddings.py --csv verticalizer/data/us_labeled.csv
