# src/verticalizer/embeddings/cache.py
import hashlib
import os
import orjson
import tempfile
import shutil

CACHE_DIR = os.environ.get("EMB_CACHE_DIR", ".embcache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _key(text: str, model: str) -> str:
    s = (model or "") + "\n" + (text or "")
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")

def get_cached(text: str, model: str):
    path = _key(text, model)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    return None

def set_cached(text: str, model: str, vec):
    path = _key(text, model)
    td = tempfile.mkdtemp(prefix="embcache-")
    try:
        tmp = os.path.join(td, "vec.json")
        with open(tmp, "wb") as f:
            f.write(orjson.dumps(vec))
        shutil.move(tmp, path)
    finally:
        shutil.rmtree(td, ignore_errors=True)