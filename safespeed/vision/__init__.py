"""SafeSpeed vision model: road attributes, invited speed and harm from street images.

    from safespeed import vision
    images = vision.read_images_csv("data/vision_sample/images.csv")
    segments = vision.read_segments_csv("data/vision_sample/segments.csv")
    emb = vision.load_embeddings("data/vision_sample/clip_embeddings.npz")
    dino = vision.load_dinov3_features("data/vision_sample/dinov3_features.npz")
    pred = vision.predict_segments(images, segments, emb,
                                   detections="data/vision_sample/detections.jsonl",
                                   dinov3=dino)

Thailand only (the model was trained on the Thai network). The chain:

  1. image filter     captured 2010 or later, within 40 m of the segment's 500 m
                      point, camera along the road (panoramas always kept)
  2. CLIP embedding   ViT-L/14, one L2-normalised 768-d vector per image. Cached
                      embeddings are used; `embed.py` recomputes them from pixels
                      (optional: torch + transformers + a 1.7 GB download)
  3. probes           11 ThaiRAP attribute probes (non-commercial weights, own
                      folder), 2 OSM probes (divided, lane count), plus
                      Mapillary's own detections (people, riders, footpaths,
                      crossings, traffic lights)
  4. aggregation      per-segment shares, `_any` flags, counts, and the mean
                      embedding on 32 principal components -> v_* columns
  5. rung             the survivable speed the pictured road allows
                      (vis_safe_speed, vis_crash_type_key)
  6. speed model      vis_pred_v50 / vis_pred_v85, exact factor attribution
                      (vis_speed_attr_*), expected harm on the rung's injury
                      curve (vis_harm)
  7. classifier       optional, only when DINOv3 features are passed: the
                      DINOv3 ThaiRAP classifier (non-commercial weights, own
                      folder) gives per-image iRAP code probabilities, averaged
                      per segment -> a_* columns (the inputs of the ensemble's
                      optional member gbm_c2g_vision_irap; the speed model above
                      does not use them)

Inputs are checked first; every problem found is listed in one VisionInputError.
No priority score is computed here: this is the model only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import aggregate as A
from .classifier import DinoV3Classifier, segment_means
from .probes import OSMProbes, ThaiRAPProbes
from .rung import vision_rung
from .speed import VisionSpeedModel

WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "weights"


class VisionInputError(ValueError):
    """The vision inputs cannot be used; the message says what to fix."""


_cache: dict = {}


def load(weights_dir: Path = WEIGHTS_DIR) -> dict:
    """Load the vision weights once. The DINOv3 classifier's heads are optional:
    without dinov3_heads.npz everything else runs."""
    key = str(weights_dir)
    if key not in _cache:
        wd = Path(weights_dir)
        vd = wd / "vision"
        cfg = json.loads((vd / "vision_config.json").read_text())
        pca = np.load(vd / "pca.npz", allow_pickle=False)
        tr = ThaiRAPProbes(wd / "vision_thairap" / "thairap_probes.npz")
        dn_path = wd / "vision_thairap" / "dinov3_heads.npz"
        dn = DinoV3Classifier(dn_path) if dn_path.exists() else None
        _cache[key] = {
            "cfg": cfg, "thairap": tr, "dinov3": dn, "osm": OSMProbes(vd / "osm_probes.npz"),
            "pca_mean": pca["mean"], "pca_components": pca["components"],
            "speed": VisionSpeedModel(vd / "vision_speed_trees.json"),
            "curves": json.loads((wd / "injury_curves.json").read_text()),
        }
    return _cache[key]


# ---------------------------------------------------------------- readers
def read_images_csv(path) -> pd.DataFrame:
    """One row per (segment point, image): image_id, seg_id, and the columns the
    filter uses (captured_at or year, distance_m, compass_angle, road_bearing,
    heading_diff, is_pano) plus osm_fold (blank outside the training network)."""
    return pd.read_csv(path, float_precision="round_trip",
                       dtype={"image_id": str, "seg_id": str, "key": str})


def read_segments_csv(path) -> pd.DataFrame:
    """One row per segment: id, country (TH), v_n_points (500 m points on it)."""
    return pd.read_csv(path, float_precision="round_trip", dtype={"id": str, "country": str})


def load_embeddings(path) -> tuple[np.ndarray, np.ndarray]:
    """(image ids, float32 embeddings) from an npz with `ids` and `emb`."""
    z = np.load(path, allow_pickle=False)
    return z["ids"].astype(str), z["emb"].astype(np.float32)


def load_dinov3_features(path) -> tuple[np.ndarray, np.ndarray]:
    """(image ids, float32 DINOv3 features) from an npz with `ids` and `X` (float16,
    as SafeSpeed's full build cached them)."""
    z = np.load(path, allow_pickle=False)
    return z["ids"].astype(str), z["X"].astype(np.float32)


def filter_images(images: pd.DataFrame, weights_dir: Path = WEIGHTS_DIR) -> pd.DataFrame:
    """The images the model uses (date, distance and heading filter), as
    SafeSpeed's full build selected them. Embed these rows, not the unfiltered table."""
    return A.filter_images(images, load(weights_dir)["cfg"])[0]


# ---------------------------------------------------------------- per image
def predict_images(images: pd.DataFrame, embeddings, detections=None,
                   weights_dir: Path = WEIGHTS_DIR, dinov3=None):
    """Per-image outputs. Returns (per-image table, projected embeddings
    for every row of `embeddings`, embedding ids).

    Every row of the embedding matrix is run through the probes and the PCA in
    one call (as SafeSpeed's full build does), then the images in `images` are selected.
    `dinov3` = (ids, features) adds the classifier's float32 a_* probabilities.
    """
    W = load(weights_dir)
    cfg = W["cfg"]
    ids, E = embeddings
    ids = np.asarray(ids).astype(str)
    E = np.asarray(E, dtype=np.float32)
    if E.ndim != 2 or E.shape[1] != cfg["embedding"]["dim"] or len(ids) != len(E):
        raise VisionInputError(f"embeddings must be ({len(ids)} x {cfg['embedding']['dim']}); "
                               f"got {E.shape}")
    need = pd.unique(images["image_id"].astype(str))
    missing = sorted(set(need) - set(ids))
    if missing:
        raise VisionInputError(f"{len(missing)} images have no embedding (e.g. "
                               f"{', '.join(missing[:5])}); compute them with "
                               "run_vision.py --embed-to NEW.npz --image-dir DIR "
                               "(needs torch + transformers)")

    # ThaiRAP probes -> codes -> features
    codes = pd.DataFrame(W["thairap"].codes(E))
    codes.insert(0, "image_id", ids)
    feats = A.thairap_features(codes, cfg)
    # detections
    det_cols = list(cfg["detections"]["features"])
    if detections is None:
        det = pd.DataFrame(columns=["image_id", *det_cols])
    elif isinstance(detections, pd.DataFrame):
        det = detections
    else:
        det = A.read_detections(detections, ids, cfg)
    feats = feats.merge(det, on="image_id", how="left")
    # OSM probes, fold from the image's first row in `images`
    fold_of = {}
    if "osm_fold" in images.columns:
        f = images.drop_duplicates("image_id").set_index("image_id")["osm_fold"]
        fold_of = pd.to_numeric(f, errors="coerce").to_dict()
    fold = np.array([fold_of.get(i, np.nan) for i in ids], dtype="float64")
    fold = np.where(np.isfinite(fold), fold, -1).astype(int)
    p_div = W["osm"].predict(E, fold, "divided")
    lanes = W["osm"].predict(E, fold, "lanes")
    osm = pd.DataFrame({"image_id": ids, "p_divided": p_div, "v_lanes_osmprobe": lanes})
    feats = feats.merge(osm, on="image_id", how="left")
    feats["v_divided"] = (feats.p_divided >= 0.5).astype(float).where(feats.p_divided.notna())
    feats["v_undivided"] = 1.0 - feats["v_divided"]
    feats["v_lanes"] = feats.pop("v_lanes_osmprobe")
    feats["osm_model"] = np.where(fold >= 0, [f"fold {k}" for k in fold], "all-data refit")
    for a in W["thairap"].attributes:
        feats[f"thairap_{a}"] = codes[a].to_numpy()
    if W["dinov3"] is not None and dinov3 is not None:
        feats = feats.merge(classify_images(images, dinov3, W["dinov3"]), on="image_id",
                            how="left")
    Z = A.pca_project(E, W["pca_mean"], W["pca_components"])
    return feats, Z, ids


def check_inputs(images: pd.DataFrame, segments: pd.DataFrame | None, country: str) -> None:
    """Raise VisionInputError listing every problem in the images and segments tables."""
    problems = []

    def blank(s: pd.Series) -> pd.Series:
        return s.isna() | s.astype(str).str.strip().eq("")

    def rows(s: pd.Series, mask) -> str:
        v = [str(x) for x in s[mask].tolist()]
        return ", ".join(v[:5]) + (f" and {len(v) - 5} more" if len(v) > 5 else "")

    miss = [c for c in ("image_id", "seg_id") if c not in images.columns]
    if "year" not in images.columns and "captured_at" not in images.columns:
        miss.append("captured_at (or year)")
    if miss:
        problems.append(f"images lack the column(s) {', '.join(miss)} (see COLUMNS.md)")
    for c in ("image_id", "seg_id"):
        if c in images.columns and blank(images[c]).any():
            problems.append(f"images: {c} is blank on {int(blank(images[c]).sum())} row(s)")
    key = images["image_id"].astype(str) if "image_id" in images.columns else \
        pd.Series([f"row {i + 1}" for i in range(len(images))], index=images.index)
    for c in ("captured_at", "year", "distance_m", "compass_angle", "road_bearing",
              "heading_diff", "osm_fold"):
        if c in images.columns:
            v = pd.to_numeric(images[c], errors="coerce")
            text = ~blank(images[c]) & v.isna()
            if text.any():
                problems.append(f"images: {c} has values that are not numbers, e.g. "
                                f"'{images[c][text].iloc[0]}' (images {rows(key, text)})")
    if "is_pano" in images.columns:
        try:
            A.parse_bool(images["is_pano"], "is_pano")
        except ValueError as e:
            problems.append(f"images: {e}")
    if segments is not None:
        if "id" not in segments.columns:
            problems.append("segments lack the column id")
        else:
            sid = segments["id"].astype(str).str.strip()
            if blank(segments["id"]).any():
                problems.append(f"segments: id is blank on {int(blank(segments['id']).sum())} row(s)")
            dup = sid.duplicated() & ~blank(segments["id"])
            if dup.any():
                problems.append(f"segments: id repeated ({rows(sid, dup)})")
            if "country" in segments.columns:
                ctry = segments["country"].astype(str).str.strip().str.upper()
                bad = ctry.ne(country) | blank(segments["country"])
                if bad.any():
                    seen = sorted({"(blank)" if b else v for v, b in
                                   zip(ctry[bad], blank(segments["country"])[bad])})
                    problems.append(f"segments: country must be {country}; got {', '.join(seen)} "
                                    f"({rows(sid, bad)}). The vision model is fitted to "
                                    "Thailand only.")
            if "v_n_points" in segments.columns:
                v = pd.to_numeric(segments["v_n_points"], errors="coerce")
                bad = v.isna() | (v < 0) | (v != np.round(v))
                if bad.any():
                    problems.append(f"segments: v_n_points must be a whole number, 0 or more "
                                    f"({rows(sid, bad)})")
            if "seg_id" in images.columns:
                orphan = ~images["seg_id"].astype(str).str.strip().isin(set(sid)) & \
                    ~blank(images["seg_id"])
                if orphan.any():
                    problems.append("images: seg_id not in the segments table ("
                                    + rows(images["seg_id"].astype(str), orphan) + ")")
    if problems:
        raise VisionInputError("the vision inputs cannot be used:\n  - " + "\n  - ".join(problems))


def classify_images(images: pd.DataFrame, dinov3, clf: DinoV3Classifier) -> pd.DataFrame:
    """DINOv3 classifier probabilities (design-role a_* columns, float32) for every
    row of the feature matrix, as SafeSpeed's full build ran it; one row per image id."""
    dids, D = dinov3
    dids = np.asarray(dids).astype(str)
    D = np.asarray(D, dtype=np.float32)
    if D.ndim != 2 or D.shape[1] != clf.feature_dim or len(dids) != len(D):
        raise VisionInputError(f"DINOv3 features must be ({len(dids)} x {clf.feature_dim}); "
                               f"got {D.shape}")
    need = pd.unique(images["image_id"].astype(str))
    missing = sorted(set(need) - set(dids))
    if missing:
        raise VisionInputError(f"{len(missing)} images have no DINOv3 features (e.g. "
                               f"{', '.join(missing[:5])}); compute them with "
                               "run_vision.py --embed-to NEW.npz --image-dir DIR --with-dinov3 "
                               "(needs torch and timm), or run without the classifier (leave out "
                               "--with-dinov3)")
    out = clf.image_scores(D)
    out.insert(0, "image_id", dids)
    return out.drop_duplicates("image_id")


# ---------------------------------------------------------------- per segment
def output_columns(cfg: dict, groups, classifier: DinoV3Classifier | None = None) -> list:
    col = cfg["columns"]
    return (["id", "country"] + col["ref"] + col["design"] + col["emb"] + col["meta"]
            + ["vis_pred_v50", "vis_pred_v85", "vis_harm"]
            + [f"vis_speed_attr_{g}" for g in groups]
            + ["vis_speed_attr_base_kmh", "vis_speed_top_factor"]
            + ["vis_safe_speed", "vis_crash_type_key", "vis_safe_speed_reason",
               "vis_motorway_like", "vis_conflict_head_on", "vis_conflict_side",
               "vis_ped_evidence", "vis_status"]
            + (classifier.columns() if classifier is not None else []))


def predict_segments(images: pd.DataFrame, segments: pd.DataFrame | None, embeddings,
                     detections=None, weights_dir: Path = WEIGHTS_DIR,
                     apply_filter: bool = True, return_images: bool = False, dinov3=None):
    """Per-segment vision features, rung, invited speeds, attribution and harm.

    images      one row per (segment point, image); see read_images_csv
    segments    one row per segment: id, country, v_n_points. None: every seg_id
                in `images`, country TH, v_n_points unknown (coverage blank)
    embeddings  (ids, float32 matrix), e.g. load_embeddings(...)
    detections  path to a Mapillary detections .jsonl, a per-image table, or None
                (then every VRU/presence feature is blank and the motorway rung
                can never be reached)
    dinov3      (ids, float32 DINOv3 features), e.g. load_dinov3_features(...), to add
                the classifier's a_* columns; None (the default) leaves them out.
    """
    W = load(weights_dir)
    cfg = W["cfg"]
    check_inputs(images, segments, W["speed"].country)
    if dinov3 is not None and W["dinov3"] is None:
        raise VisionInputError("DINOv3 features were given, but weights/vision_thairap/"
                               "dinov3_heads.npz is missing")
    img = images.copy()
    img["image_id"] = img["image_id"].astype(str)
    img["seg_id"] = img["seg_id"].astype(str)
    if "year" not in img.columns:
        if "captured_at" not in img.columns:
            raise VisionInputError("images need `year` or `captured_at`")
        img["year"] = A.capture_year(img["captured_at"])
    if apply_filter:
        img, _ = A.filter_images(img, cfg)
    if segments is None:
        segments = pd.DataFrame({"id": pd.unique(img["seg_id"]), "country": "TH"})
    seg_in = segments.copy()
    seg_in["id"] = seg_in["id"].astype(str)
    if "country" not in seg_in.columns:
        seg_in["country"] = "TH"

    feats, Z, ids = predict_images(img, embeddings, detections, weights_dir, dinov3=dinov3)
    keep = [c for c in feats.columns if c.startswith("v_") or c == "image_id"]
    emb = A.segment_embeddings(img, ids, Z, cfg["columns"]["emb"])
    pts = (pd.to_numeric(seg_in.set_index("id")["v_n_points"], errors="coerce")
           if "v_n_points" in seg_in.columns else pd.Series(dtype=float))
    agg = A.aggregate(feats[keep], img, pts, emb, cfg).set_index("id")

    seg = seg_in[["id", "country"]].set_index("id")
    out = seg.join(agg, how="left")
    for c in ("v_n_images", "v_n_forward", "v_n_evidence"):
        out[c] = out[c].fillna(0).astype(int)
    out["v_n_points"] = (pts.reindex(out.index).fillna(0).astype(int) if len(pts)
                         else out["v_n_points"].fillna(0).astype(int))
    for c in cfg["columns"]["emb"]:
        out[c] = out[c].astype("float64")          # exact; keeps CSV round trips exact
    out = out.reset_index()
    rung = vision_rung(out, cfg)
    speed = W["speed"].run(out, rung["vis_crash_type_key"].to_numpy(), W["curves"])
    res = pd.concat([out, speed, rung], axis=1)
    clf = W["dinov3"] if dinov3 is not None else None
    if clf is not None:
        acols = clf.columns()
        a = pd.DataFrame(np.nan, index=res.index, columns=acols)
        if acols[0] in feats.columns:
            # the segment's (point, image) rows in input order (no merge: its row
            # order differs between pandas versions, and the float32 sum is ordered)
            f = feats.drop_duplicates("image_id")
            at = pd.Index(f["image_id"].astype(str)).get_indexer(img["image_id"].astype(str))
            ok = at >= 0
            sid, M = segment_means(img["seg_id"].to_numpy()[ok],
                                   f[acols].to_numpy(np.float32)[at[ok]])
            pos = pd.Series(np.arange(len(sid)), index=sid)
            hit = res["id"].isin(pos.index).to_numpy()
            a.loc[hit, acols] = M[pos.loc[res.loc[hit, "id"]].to_numpy()].astype("float64")
        res = pd.concat([res, a], axis=1)       # float64 of the float32 means: exact in CSV
    res = res[output_columns(cfg, W["speed"].groups, clf)]
    if return_images:
        return res, img.merge(feats, on="image_id", how="left")
    return res
