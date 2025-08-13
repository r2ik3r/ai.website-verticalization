# Embedding Module

Purpose
- Generate embeddings for latest crawled content.

Input
- CSV with website column; model name.

Output
- embeddings table rows; .npy in S3 (optional).

Command
- poetry run verticalizer embed --in data/sites.csv --model models/text-embedding-004

Cost-safety
- Controlled by env: GEMINI_EMB_DRYRUN, GEMINI_EMB_MAX_CALLS, GEMINI_EMB_RATE_LIMIT.
