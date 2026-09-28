"""Street images -> per-image features -> per-segment vision features.

Faithful port of the SafeSpeed pipeline (app/vision/worklist.py filters,
aggregate.py, detections.py). The rules and tables live in
weights/vision/vision_config.json.

Per image
  * ThaiRAP probe codes -> v_roadside_dist, v_roadside_severe,
    v_roadside_barrier, v_shoulder, v_access, v_skid, v_unsealed, v_lighting,
    v_calming, v_service_road
  * OSM probes -> v_divided (P >= 0.5), v_undivided = 1 - v_divided, v_lanes
  * Mapillary detections -> v_ped_seen / v_ped_count, v_moto_seen /
    v_moto_count, v_bicycle_seen, v_sidewalk, v_crossing, v_intersection.
    An image Mapillary never ran its detector on is NaN, never 0.

Per segment
  * each feature is the mean over the segment's (point, image) rows. Points are
    500 m apart, so the mean is a length share. An image nearest to two points
    counts twice.
  * presence flags also get `_any` (seen in at least one image);
  * v_n_images / v_n_forward (unique images), v_n_evidence (images with
    detections), v_n_points (500 m points of the segment, with or without an
    image), v_coverage_frac, v_capture_year (median);
  * v_emb_00..31: the mean over the same rows of the images' CLIP embeddings
    projected on 32 principal components.

The PCA projection repeats scikit-learn's float32 arithmetic
(x @ C^T - mean @ C^T). A float32 matrix product is rounded differently for
different matrix sizes, so v_emb_* reproduce the pipeline bit for bit only when
the same set of images is projected in one call; otherwise they agree to float32
precision (about 1e-7).
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ filters
def axial_diff(a: float, b: float) -> float:
    """Smallest angle between two headings, ignoring travel direction (0-90)."""
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return min(d, 180.0 - d)


def capture_year(captured_at: pd.Series) -> pd.Series:
    """Capture year from a Mapillary `captured_at` (milliseconds since 1970)."""
    return pd.to_datetime(pd.to_numeric(captured_at, errors="coerce"),
                          unit="ms", errors="coerce").dt.year


_TRUE = {"true", "1", "1.0", "t", "yes", "y"}
_FALSE = {"false", "0", "0.0", "f", "no", "n", "", "nan", "none"}


def parse_bool(col: pd.Series, name: str) -> pd.Series:
    """True/False column from bools, 1/0 or yes/no text (blank = False).
    Anything else raises ValueError, so a panorama is never dropped silently."""
    s = col.astype(str).str.strip().str.lower()
    bad = sorted(set(s[~s.isin(_TRUE | _FALSE)]))
    if bad:
        raise ValueError(f"{name}: expected true/false, 1/0 or yes/no; got {bad[:5]}")
    return s.isin(_TRUE)


def filter_images(images: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Keep images an iRAP coder would use: captured in or after `min_capture_year`,
    within `max_distance_m` of the segment point, and looking along the road
    (camera within `max_heading_diff_deg` of the road in either direction;
    panoramas always kept; kept when the road bearing is unknown). Missing
    columns skip their filter. Returns (kept rows, counts)."""
    f = cfg["image_filter"]
    m = images.copy()
    n0 = len(m)
    if "year" not in m.columns and "captured_at" in m.columns:
        m["year"] = capture_year(m["captured_at"])
    if "year" in m.columns:
        m = m[pd.to_numeric(m["year"], errors="coerce").ge(f["min_capture_year"])]
    n1 = len(m)
    if "distance_m" in m.columns:
        m = m[pd.to_numeric(m["distance_m"], errors="coerce").le(f["max_distance_m"])]
    n2 = len(m)
    pano = (parse_bool(m["is_pano"], "is_pano") if "is_pano" in m.columns
            else pd.Series(False, index=m.index))
    if "road_bearing" in m.columns:
        rb = pd.to_numeric(m["road_bearing"], errors="coerce")
        if "heading_diff" in m.columns:
            hd = pd.to_numeric(m["heading_diff"], errors="coerce")
        else:
            ca = pd.to_numeric(m.get("compass_angle"), errors="coerce")
            hd = pd.Series([axial_diff(c, r) if np.isfinite(r) and np.isfinite(c) else np.nan
                            for c, r in zip(ca, rb)], index=m.index)
        m = m[pano | hd.le(f["max_heading_diff_deg"]) | rb.isna()]
    elif "heading_diff" in m.columns:       # no road bearing, but the angle is given
        hd = pd.to_numeric(m["heading_diff"], errors="coerce")
        m = m[pano | hd.le(f["max_heading_diff_deg"]) | hd.isna()]
    counts = {"rows_in": n0, "after_date": n1, "after_distance": n2, "after_heading": len(m)}
    return m, counts


# ------------------------------------------------------------ per image
def _rule(rule: dict, codes: dict) -> float:
    op = rule["op"]
    raw = [codes.get(a) for a in rule["attrs"]]
    if op == "mapped":
        v = raw[0]
        return np.nan if v is None else rule["map"].get(v, np.nan)
    if op == "in":
        v = raw[0]
        return np.nan if v is None else float(v in rule["codes"])
    if op == "min_mapped":
        xs = [rule["map"][v] for v in raw if v in rule["map"]]
        return float(min(xs)) if xs else np.nan
    if op == "mean_mapped":
        xs = [rule["map"][v] for v in raw if v in rule["map"]]
        return float(np.mean(xs)) if xs else np.nan
    if op == "any_in":
        xs = [v for v in raw if v is not None]
        return float(any(v in rule["codes"] for v in xs)) if xs else np.nan
    raise ValueError(f"unknown feature rule {op!r}")


def compile_rules(cfg: dict) -> dict:
    """Feature rules with integer code keys (JSON stores them as strings)."""
    out = {}
    for name, r in cfg["thairap_features"].items():
        r = dict(r)
        if "map" in r:
            r["map"] = {int(k): v for k, v in r["map"].items()}
        if "codes" in r:
            r["codes"] = set(int(c) for c in r["codes"])
        out[name] = r
    return out


def thairap_features(codes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-image features from ThaiRAP probe codes (one column per attribute).
    Attributes absent from `codes` count as not coded (NaN features)."""
    rules = compile_rules(cfg)
    attrs = [c for c in codes.columns if c != "image_id"]
    rows = []
    for rec in codes[attrs].to_dict("records"):
        c = {k: (int(v) if pd.notna(v) else None) for k, v in rec.items()}
        rows.append({name: _rule(r, c) for name, r in rules.items()})
    out = pd.DataFrame(rows, columns=list(rules), index=codes.index, dtype="float64")
    out.insert(0, "image_id", codes["image_id"].to_numpy())
    return out


def read_detections(path, image_ids=None, cfg: dict | None = None) -> pd.DataFrame:
    """Mapillary detections (one JSON line per image: image_id, n, values) ->
    per-image presence features. Images without a line, or with n = 0, get no
    row (NaN after the merge): no detector run is not evidence of absence."""
    classes = {k: tuple(v) for k, v in cfg["detections"]["classes"].items()}
    feats = cfg["detections"]["features"]
    keep = None if image_ids is None else set(map(str, image_ids))
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if keep is not None and str(r["image_id"]) not in keep:
                continue
            if not r.get("n") or not r.get("values"):
                continue
            c = Counter(r["values"])
            n = {k: sum(v for key, v in c.items() if key.startswith(p)) for k, p in classes.items()}
            row = {"image_id": str(r["image_id"])}
            for name, (op, cls) in feats.items():
                row[name] = float(n[cls] > 0) if op == "seen" else float(n[cls])
            rows.append(row)
    return pd.DataFrame(rows, columns=["image_id", *feats])


# ------------------------------------------------------------ embeddings
def pca_project(E: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """scikit-learn PCA.transform, same float32 arithmetic."""
    Z = E @ components.T
    Z -= mean.reshape(1, -1) @ components.T
    return Z


def segment_embeddings(images: pd.DataFrame, ids: np.ndarray, Z: np.ndarray,
                       cols: list) -> pd.DataFrame:
    """Mean projected embedding over each segment's (point, image) rows."""
    pos = {i: k for k, i in enumerate(ids)}
    w = images[images["image_id"].isin(pos)]
    rows, keys = [], []
    for sid, g in w.groupby("seg_id"):
        rows.append(Z[[pos[i] for i in g["image_id"]]].mean(axis=0))
        keys.append(sid)
    if not rows:
        return pd.DataFrame(columns=cols, dtype="float32")
    return pd.DataFrame(np.vstack(rows), index=keys, columns=cols)


# ------------------------------------------------------------ per segment
def aggregate(feats: pd.DataFrame, images: pd.DataFrame, points_per_seg: pd.Series,
              seg_emb: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-segment vision features (the pipeline's `aggregate`)."""
    col = cfg["columns"]
    w = images[["seg_id", "image_id", "year"]].merge(feats, on="image_id", how="inner")
    n_img = w.groupby("seg_id").image_id.nunique().rename("v_n_images")
    fw = w                                   # every kept image is a forward view
    share_cols = [c for c in [c for c in col["ref"] if not c.endswith("_any")] + col["design"]
                  if c in fw.columns]
    share = fw.groupby("seg_id")[share_cols].mean()
    anyc = [c for c in col["any"] if c in fw.columns]
    anyv = fw.groupby("seg_id")[anyc].max().add_suffix("_any")
    out = share.join(anyv, how="outer")
    out = out.join(n_img, how="outer")
    out["v_n_forward"] = fw.groupby("seg_id").image_id.nunique()
    out["v_n_forward"] = out.v_n_forward.fillna(0).astype(int)
    ev = fw[fw["v_ped_seen"].notna()] if "v_ped_seen" in fw.columns else fw.iloc[:0]
    out["v_n_evidence"] = ev.groupby("seg_id").image_id.nunique()
    out["v_n_evidence"] = out.v_n_evidence.reindex(out.index).fillna(0).astype(int)
    out["v_n_points"] = points_per_seg.reindex(out.index).fillna(0).astype(int)
    out["v_coverage_frac"] = (out.v_n_forward / out.v_n_points.replace(0, np.nan)).clip(0, 1).round(3)
    out["v_capture_year"] = w.groupby("seg_id").year.median()
    out["v_model"] = cfg["v_model"] if len(w) else None
    out["v_prompt_hash"] = cfg["v_prompt_hash"] if len(w) else None
    out = out.join(seg_emb, how="left")
    out.index.name = "id"
    cols = col["ref"] + col["design"] + col["emb"] + col["meta"]
    return out.reindex(columns=cols).reset_index()
