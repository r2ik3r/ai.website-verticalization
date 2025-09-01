# src/verticalizer/pipeline/postprocess.py

from typing import List, Dict
import numpy as np

def enforce_hierarchy(probs: np.ndarray, classes: List[str], graph: Dict[str, List[str]], min_parent_prob: float = 1e-6) -> np.ndarray:
    """
    If a child has nonzero prob, ensure its ancestors have at least min_parent_prob.
    Keeps calibration shape; only raises tiny floor on parents to preserve consistency.
    """
    out = probs.copy()
    idx = {c: i for i, c in enumerate(classes)}
    # Build reverse edges (child -> parent)
    child2parent: Dict[str, str] = {}
    for p, children in graph.items():
        for ch in children:
            child2parent[ch] = p
    # Iterate until fixed point or depth
    for _ in range(5):
        changed = False
        for ch, p in child2parent.items():
            if ch not in idx or p not in idx:
                continue
            ci, pi = idx[ch], idx[p]
            # If child has meaningful mass, ensure parent floor
            child_mass = out[:, ci]
            parent_mass = out[:, pi]
            need = (child_mass > 0).astype(np.float32) * min_parent_prob
            new_parent = np.maximum(parent_mass, need)
            if np.any(new_parent > parent_mass):
                out[:, pi] = new_parent
                changed = True
        if not changed:
            break
    return out

def add_parents_to_topk(topk_ids: List[str], graph: Dict[str, List[str]]) -> List[str]:
    """
    Add missing parents for any selected child to keep human‑readable top‑k consistent.
    """
    out = list(topk_ids)
    existing = set(out)
    def parent_of(x):
        for p, children in graph.items():
            if x in children:
                return p
        return None
    for x in list(out):
        p = parent_of(x)
        while p and p not in existing:
            out.append(p)
            existing.add(p)
            p = parent_of(p)
    # Deduplicate preserving order
    seen = set()
    keep = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        keep.append(x)
    return keep