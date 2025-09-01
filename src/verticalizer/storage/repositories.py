# src/verticalizer/storage/repositories.py
from typing import List, Optional, Dict
from sqlalchemy import text
from .db import engine

def create_tables_if_missing():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS sites (
            site TEXT PRIMARY KEY,
            firstseen TIMESTAMP DEFAULT NOW(),
            lastcrawledat TIMESTAMP,
            lasthash TEXT,
            notes TEXT
        )"""))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS crawls (
            id BIGSERIAL PRIMARY KEY,
            site TEXT REFERENCES sites(site),
            url TEXT,
            fetchedat TIMESTAMP DEFAULT NOW(),
            httpstatus INTEGER,
            contenthash TEXT,
            textexcerpt TEXT,
            textfullref TEXT,
            lang TEXT,
            source TEXT,
            crawlstatus TEXT
        )"""))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id BIGSERIAL PRIMARY KEY,
            site TEXT REFERENCES sites(site),
            modelname TEXT,
            dim INTEGER,
            createdat TIMESTAMP DEFAULT NOW(),
            shatext TEXT,
            vectorref TEXT,
            vectorlen INTEGER
        )"""))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS models (
            id BIGSERIAL PRIMARY KEY,
            geo TEXT,
            version TEXT,
            pathmodel TEXT,
            pathcalib TEXT,
            createdat TIMESTAMP DEFAULT NOW(),
            configjson TEXT
        )"""))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS predictions (
            id BIGSERIAL PRIMARY KEY,
            site TEXT,
            modelversion TEXT,
            createdat TIMESTAMP DEFAULT NOW(),
            topkjson TEXT,
            rawjson TEXT
        )"""))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS evalreports (
            id BIGSERIAL PRIMARY KEY,
            modelversion TEXT,
            createdat TIMESTAMP DEFAULT NOW(),
            metricsjson TEXT
        )"""))

def upsert_site(site: str, contenthash: Optional[str] = None):
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO sites(site, firstseen, lastcrawledat, lasthash)
                    VALUES (:site, NOW(), NULL, :h)
                    ON CONFLICT(site) DO NOTHING"""),
            {"site": site, "h": contenthash},
        )

def record_crawl(site: str, url: str, httpstatus: int, contenthash: str,
                 textexcerpt: str, textfullref: str, lang: str, source: str, status: str):
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE sites SET lastcrawledat=NOW(), lasthash=:h WHERE site=:site"),
            {"h": contenthash, "site": site},
        )
        conn.execute(
            text("""INSERT INTO crawls(site, url, httpstatus, contenthash, textexcerpt,
                                       textfullref, lang, source, crawlstatus)
                    VALUES (:site, :url, :st, :h, :ex, :ref, :lang, :src, :cst)"""),
            {"site": site, "url": url, "st": httpstatus, "h": contenthash, "ex": textexcerpt,
             "ref": textfullref, "lang": lang, "src": source, "cst": status},
        )

def latest_text_for_site_batch(sites: List[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {s: None for s in sites}
    if not sites:
        return out
    with engine.connect() as conn:
        q = text("""
            SELECT DISTINCT ON (c.site) c.site, c.textexcerpt
            FROM crawls c
            WHERE c.site = ANY(:sites)
            ORDER BY c.site, c.fetchedat DESC
        """)
        rows = conn.execute(q, {"sites": sites}).all()
        for site, textexcerpt in rows:
            out[site] = textexcerpt
    return out

def record_embedding(site: str, modelname: str, dim: int, shatext: str, vectorref: str, vectorlen: int):
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO embeddings(site, modelname, dim, shatext, vectorref, vectorlen)
                    VALUES (:site, :modelname, :dim, :shatext, :vectorref, :vectorlen)"""),
            {"site": site, "modelname": modelname, "dim": dim, "shatext": shatext,
             "vectorref": vectorref, "vectorlen": vectorlen},
        )

def save_model_version(geo: str, version: str, pathmodel: str, pathcalib: str, configjson: dict):
    import json as _json
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO models(geo, version, pathmodel, pathcalib, configjson)
                    VALUES (:geo, :version, :pathmodel, :pathcalib, :cfg)"""),
            {"geo": geo, "version": version, "pathmodel": pathmodel, "pathcalib": pathcalib,
             "cfg": _json.dumps(configjson or {})},
        )

def record_prediction(site: str, modelversion: str, topkjson: dict, rawjson: dict):
    import json as _json
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO predictions(site, modelversion, topkjson, rawjson)
                    VALUES (:site, :mv, :topk, :raw)"""),
            {"site": site, "mv": modelversion, "topk": _json.dumps(topkjson), "raw": _json.dumps(rawjson)},
        )

def record_eval(modelversion: str, metricsjson: dict):
    import json as _json
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO evalreports(modelversion, metricsjson)
                    VALUES (:mv, :mj)"""),
            {"mv": modelversion, "mj": _json.dumps(metricsjson or {})},
        )