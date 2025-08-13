from typing import Optional, List, Dict
from sqlalchemy import text
from .db import engine

def create_tables_if_missing():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS sites (
            site TEXT PRIMARY KEY,
            first_seen TIMESTAMP,
            last_crawled_at TIMESTAMP,
            last_hash TEXT,
            notes TEXT
        );"""))
        conn.execute(text("""
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
        );"""))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id BIGSERIAL PRIMARY KEY,
            site TEXT REFERENCES sites(site),
            model_name TEXT,
            dim INTEGER,
            created_at TIMESTAMP DEFAULT NOW(),
            sha_text TEXT,
            vector_ref TEXT,
            vector_len INTEGER
        );"""))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS models (
            id BIGSERIAL PRIMARY KEY,
            geo TEXT,
            version TEXT,
            path_model TEXT,
            path_calib TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            config_json TEXT
        );"""))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS predictions (
            id BIGSERIAL PRIMARY KEY,
            site TEXT,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            topk_json TEXT,
            raw_json TEXT
        );"""))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS eval_reports (
            id BIGSERIAL PRIMARY KEY,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            metrics_json TEXT
        );"""))

def upsert_site(site: str, content_hash: Optional[str] = None):
    with engine.begin() as conn:
        conn.execute(text("""
        INSERT INTO sites(site, first_seen, last_crawled_at, last_hash)
        VALUES (:site, NOW(), NULL, :h)
        ON CONFLICT (site) DO NOTHING
        """), dict(site=site, h=content_hash))

def record_crawl(site: str, url: str, http_status: int, content_hash: str,
                 text_excerpt: str, text_full_ref: str, lang: str, source: str, status: str):
    with engine.begin() as conn:
        conn.execute(text("""UPDATE sites SET last_crawled_at=NOW(), last_hash=:h WHERE site=:site"""), dict(h=content_hash, site=site))
        conn.execute(text("""
        INSERT INTO crawls(site, url, http_status, content_hash, text_excerpt, text_full_ref, lang, source, crawl_status)
        VALUES (:site, :url, :st, :h, :ex, :ref, :lang, :src, :cst)
        """), dict(site=site, url=url, st=http_status, h=content_hash, ex=text_excerpt, ref=text_full_ref, lang=lang, src=source, cst=status))

def latest_text_for_site_batch(sites: List[str]) -> Dict[str, Optional[str]]:
    with engine.connect() as conn:
        q = text("""
        SELECT DISTINCT ON (c.site) c.site, c.text_excerpt
        FROM crawls c
        WHERE c.site = ANY(:sites)
        ORDER BY c.site, c.fetched_at DESC
        """)
        rows = conn.execute(q, dict(sites=sites)).all()
        out: Dict[str, Optional[str]] = {}
        for row in rows:
            site, text_excerpt = row, row
            out[site] = text_excerpt
        for s in sites:
            out.setdefault(s, None)
        return out

def record_embedding(site: str, model_name: str, dim: int, sha_text: str, vector_ref: str, vector_len: int):
    with engine.begin() as conn:
        conn.execute(text("""
        INSERT INTO embeddings(site, model_name, dim, sha_text, vector_ref, vector_len)
        VALUES (:site, :model_name, :dim, :sha_text, :vector_ref, :vector_len)
        """), dict(site=site, model_name=model_name, dim=dim, sha_text=sha_text, vector_ref=vector_ref, vector_len=vector_len))

def save_model_version(geo: str, version: str, path_model: str, path_calib: str, config_json: dict):
    import json as _json
    with engine.begin() as conn:
        conn.execute(text("""
        INSERT INTO models(geo, version, path_model, path_calib, config_json)
        VALUES (:geo, :version, :path_model, :path_calib, :config_json)
        """), dict(geo=geo, version=version, path_model=path_model, path_calib=path_calib, config_json=_json.dumps(config_json)))

def record_prediction(site: str, model_version: str, topk_json: dict, raw_json: dict):
    import json as _json
    with engine.begin() as conn:
        conn.execute(text("""
        INSERT INTO predictions(site, model_version, topk_json, raw_json)
        VALUES (:site, :mv, :topk, :raw)
        """), dict(site=site, mv=model_version, topk=_json.dumps(topk_json), raw=_json.dumps(raw_json)))

def record_eval(model_version: str, metrics_json: dict):
    import json as _json
    with engine.begin() as conn:
        conn.execute(text("""
        INSERT INTO eval_reports(model_version, metrics_json)
        VALUES (:mv, :mj)
        """), dict(mv=model_version, mj=_json.dumps(metrics_json)))
