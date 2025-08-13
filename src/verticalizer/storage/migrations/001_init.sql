-- Initial schema
CREATE TABLE IF NOT EXISTS sites (
  site TEXT PRIMARY KEY,
  first_seen TIMESTAMP,
  last_crawled_at TIMESTAMP,
  last_hash TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS crawls (
  id BIGSERIAL PRIMARY KEY,
  site TEXT REFERENCES sites(site),
  url TEXT,
  fetched_at TIMESTAMP DEFAULT NOW(),
  http_status INTEGER,
  content_hash TEXT,
  text_excerpt TEXT,
  text_full_ref TEXT,
  lang TEXT,
  source TEXT,
  crawl_status TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
  id BIGSERIAL PRIMARY KEY,
  site TEXT REFERENCES sites(site),
  model_name TEXT,
  dim INTEGER,
  created_at TIMESTAMP DEFAULT NOW(),
  sha_text TEXT,
  vector_ref TEXT,
  vector_len INTEGER
);

CREATE TABLE IF NOT EXISTS models (
  id BIGSERIAL PRIMARY KEY,
  geo TEXT,
  version TEXT,
  path_model TEXT,
  path_calib TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  config_json TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
  id BIGSERIAL PRIMARY KEY,
  site TEXT,
  model_version TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  topk_json TEXT,
  raw_json TEXT
);

CREATE TABLE IF NOT EXISTS eval_reports (
  id BIGSERIAL PRIMARY KEY,
  model_version TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  metrics_json TEXT
);
