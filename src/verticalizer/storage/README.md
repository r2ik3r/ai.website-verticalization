# Storage Layer

Postgres tables
- sites(site PK, first_seen, last_crawled_at, last_hash)
- crawls(id, site FK, url, fetched_at, http_status, content_hash, text_excerpt, text_full_ref, lang, source, crawl_status)
- embeddings(id, site FK, model_name, dim, created_at, sha_text, vector_ref, vector_len)
- models(id, geo, version, path_model, path_calib, created_at, config_json)
- predictions(id, site, model_version, created_at, topk_json, raw_json)
- eval_reports(id, model_version, created_at, metrics_json)

Object storage keys (optional)
- raw_html/{site}/{content_hash}.html
- embeddings/{site}/{model}/{sha_text}.npy
- models/{geo}/{version}/...
