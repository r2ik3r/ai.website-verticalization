# src/verticalizer/utils/taxonomy_versioned.py

import json
import os
from typing import Dict, Tuple, List, Optional, Set

BASEDIR = os.path.dirname(os.path.dirname(__file__))
TAXDIR = os.path.join(os.path.dirname(BASEDIR), "data", "taxonomy")

# Expected files to exist under data/taxonomy:
# v3/id_to_label.json, v3/label_to_id.json, v3/graph.json
# v2_2/id_to_label.json, v2_2/label_to_id.json, v2_2/graph.json
# Optional mapping files:
# map/v2_2_to_v3.json, map/v3_to_v2_2.json

def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_taxonomy(version: str = "v3") -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]], Dict[str, int]]:
    """
    Load IAB taxonomy by version.
    Returns:
      id2label: IAB_ID -> human label
      label2id: lowercased human label -> IAB_ID
      graph: parent_id -> list of child_ids (Tier tree)
      tier_map: IAB_ID -> tier (1,2,3,4)
    """
    vdir = "v3" if version in ("v3", "3.0", "3") else "v2_2"
    id2label = _read_json(os.path.join(TAXDIR, vdir, "id_to_label.json"))
    label2id = _read_json(os.path.join(TAXDIR, vdir, "label_to_id.json"))
    graph = _read_json(os.path.join(TAXDIR, vdir, "graph.json"))
    # Build tier map from graph
    tier_map: Dict[str, int] = {}
    roots: Set[str] = set(id2label.keys()) - {c for children in graph.values() for c in children}
    # BFS from roots to set tiers
    from collections import deque
    dq = deque()
    for r in roots:
        tier_map[r] = 1
        dq.append(r)
    while dq:
        p = dq.popleft()
        for ch in graph.get(p, []):
            tier_map[ch] = tier_map.get(p, 1) + 1
            dq.append(ch)
    # Ensure any isolated nodes have a tier
    for k in id2label.keys():
        tier_map.setdefault(k, 1)
    return id2label, label2id, graph, tier_map

def load_mapping(src_version: str, dst_version: str) -> Dict[str, List[str]]:
    """
    Load mapping from src_version to dst_version.
    Returns map: src_id -> list of dst_ids (many-to-one or one-to-many).
    """
    if src_version == dst_version:
        return {}
    mfile = "v2_2_to_v3.json" if (src_version.startswith("v2") and dst_version.startswith("v3")) else "v3_to_v2_2.json"
    path = os.path.join(TAXDIR, "map", mfile)
    if os.path.exists(path):
        return _read_json(path)
    return {}

def map_between_versions(ids: List[str], src_version: str, dst_version: str) -> List[str]:
    if src_version == dst_version:
        return list(dict.fromkeys([x.strip().upper() for x in ids if x]))
    mapping = load_mapping(src_version, dst_version)
    out: List[str] = []
    for i in ids:
        i = (i or "").strip().upper()
        if not i:
            continue
        mapped = mapping.get(i, [])
        if not mapped:
            # Fallback: keep original if unknown
            out.append(i)
        else:
            out.extend(mapped)
    # Dedup preserving order
    return list(dict.fromkeys(out))

def get_parent(child_id: str, graph: Dict[str, List[str]]) -> Optional[str]:
    for p, children in graph.items():
        if child_id in children:
            return p
    return None

def get_ancestors(node: str, graph: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    cur = node
    seen: Set[str] = set()
    while True:
        par = get_parent(cur, graph)
        if not par or par in seen:
            break
        out.append(par)
        seen.add(par)
        cur = par
    return out

def is_parent(child_id: str, parent_id: str, graph: Dict[str, List[str]]) -> bool:
    return parent_id in get_ancestors(child_id, graph)

def normalize_labels(raw: List[str], version: str = "v3") -> List[str]:
    id2label, label2id, _, _ = load_taxonomy(version)
    out: List[str] = []
    for x in raw or []:
        x = (x or "").strip()
        if not x:
            continue
        xu = x.upper()
        if xu in id2label:
            out.append(xu)
            continue
        xl = x.strip().lower()
        if xl in label2id:
            out.append(label2id[xl])
    return list(dict.fromkeys(out))