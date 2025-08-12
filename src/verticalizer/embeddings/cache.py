import hashlib
import os
import orjson

CACHE_DIR = os.environ.get("EMB_CACHE_DIR", ".emb_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _key(text: str, model: str) -> str:
    h = hashlib.sha256((model + "||" + text).encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, h + ".json")

def get_cached(text: str, model: str):
    path = _key(text, model)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    return None

def set_cached(text: str, model: str, vec):
    path = _key(text, model)
    with open(path, "wb") as f:
        f.write(orjson.dumps(vec))
