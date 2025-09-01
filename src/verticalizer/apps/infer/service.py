# src/verticalizer/apps/infer/service.py

import os
from typing import List, Optional
import pandas as pd
import numpy as np

from ...models.persistence import loadmodel
from ...models.calibration import ProbCalibrator
from ...utils.taxonomy_versioned import load_taxonomy
from ...pipeline.postprocess import enforce_hierarchy, add_parents_to_topk
from ...pipeline.ensemble import load_many_models, load_many_calibrators, apply_many_calibrators, average_probs
from ...pipeline.common import prepareembeddingsfordf
from ...pipeline.io import writejsonl

def _aggregate_group(probs_pages: np.ndarray, method: str = "mean") -> np.ndarray:
    if probs_pages.ndim != 2:
        raise ValueError("Expected 2D probs per group")
    if method == "mean":
        return probs_pages.mean(axis=0, keepdims=True)
    elif method == "softmax_mean":
        z = np.clip(probs_pages, -20, 20)
        e = np.exp(z)
        sm = e / (np.sum(e, axis=1, keepdims=True) + 1e-9)
        return sm.mean(axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown agg: {method}")

def infer_from_csv(
    incsv: str,
    modelpath: Optional[str],
    calibpath: Optional[str],
    outjsonl: str,
    topk: int = 10,
    models: Optional[List[str]] = None,
    calibs: Optional[List[str]] = None,
    iab_version: str = "v3",
    hierarchy_consistent: bool = True,
    group_col: Optional[str] = None,
    url_col: Optional[str] = None,
    page_agg: str = "mean",
    ensemble_method: str = "mean",
) -> str:
    """
    Extended inference:
      - Single or multiple models (ensemble)
      - Hierarchy consistency
      - Multi-URL per site aggregation when group_col+url_col provided
      - Versioned taxonomy
    Input CSV schema:
      - website (required)
      - Optional: url when paging per site
      - Optional: content text (if present, prepareembeddingsfordf will pick it up via crawl/embed reuse)
    """
    df = pd.read_csv(incsv)
    if "website" not in df.columns:
        raise ValueError("CSV must contain 'website' column")
    id2label, _, graph, _ = load_taxonomy(iab_version)
    classes = list(id2label.keys())

    def predict_df(dfin: pd.DataFrame) -> np.ndarray:
        X = prepareembeddingsfordf(dfin)
        if models and len(models) > 0:
            model_objs = load_many_models(models)
            cal_objs = load_many_calibrators(calibs or [None] * len(model_objs))
            raw_list = []
            for m in model_objs:
                raw = m.predict(X, verbose=0)
                if isinstance(raw, (list, tuple)):
                    raw = raw
                raw_list.append(raw)
            probs_list = apply_many_calibrators(raw_list, cal_objs)
            probs = average_probs(probs_list, method=ensemble_method)
        else:
            model_obj = loadmodel(modelpath) if modelpath else None
            raw = model_obj.predict(X, verbose=0)
            if isinstance(raw, (list, tuple)):
                raw = raw
            cal = ProbCalibrator.load(calibpath) if calibpath and os.path.exists(calibpath) else ProbCalibrator()
            probs = cal.transform(raw) if getattr(cal, "cals", None) else raw
        if hierarchy_consistent:
            probs = enforce_hierarchy(probs, classes, graph, min_parent_prob=1e-6)
        return probs

    outputs = []
    if group_col and url_col and group_col in df.columns and url_col in df.columns:
        # Page-level prediction then aggregate to site-level
        groups = df.groupby(group_col, dropna=False, sort=False)
        for site, g in groups:
            g = g.reset_index(drop=True)
            page_probs = predict_df(g)
            site_prob = _aggregate_group(page_probs, method=page_agg)  # shape (1, L)
            p = site_prob
            order = np.argsort(-p)[:max(1, topk)]
            cats = [{"id": classes[j], "label": id2label.get(classes[j], classes[j]), "prob": float(p[j])} for j in order]
            # Optionally add parents to top-k list without changing probs
            ids_only = [c["id"] for c in cats]
            ids_aug = add_parents_to_topk(ids_only, graph)
            # Rebuild cats preserving original order and appending parents at end with current probs
            seen = set(ids_only)
            for add_id in ids_aug:
                if add_id in seen:
                    continue
                j = classes.index(add_id)
                cats.append({"id": add_id, "label": id2label.get(add_id, add_id), "prob": float(p[j])})
                seen.add(add_id)
            outputs.append({"website": str(site).strip().lower(), "categories": cats})
    else:
        probs = predict_df(df)
        for i, row in df.iterrows():
            p = probs[i]
            order = np.argsort(-p)[:max(1, topk)]
            cats = [{"id": classes[j], "label": id2label.get(classes[j], classes[j]), "prob": float(p[j])} for j in order]
            ids_only = [c["id"] for c in cats]
            ids_aug = add_parents_to_topk(ids_only, graph)
            seen = set(ids_only)
            for add_id in ids_aug:
                if add_id in seen:
                    continue
                j = classes.index(add_id)
                cats.append({"id": add_id, "label": id2label.get(add_id, add_id), "prob": float(p[j])})
                seen.add(add_id)
            outputs.append({"website": str(row["website"]).strip().lower(), "categories": cats})

    writejsonl(outjsonl, outputs)
    return outjsonl