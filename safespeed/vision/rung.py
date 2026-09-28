"""The survivable-speed rung, decided from what the street images show.

Faithful port of the pipeline's app/vision/rung.py. The safe speed is the LOWEST
rung whose conflict the road allows (speeds derived from the Lubbe, Wu &
Jeppsson 2022 injury curves: people on foot 30, cyclists 40, motorcyclists 45,
side impact 50, head-on 55, motorway 100 km/h):

  vru_ped      people on foot, a footpath, a crossing or a school zone seen in
               >= ped_share of the images, or area type Urban in >= urban_share
  vru_cycle    cyclists seen in >= cycle_share of the images
  vru_ptw      motorcycles are assumed on every road that is not motorway-like
  side_impact  a signalised junction seen in any image (a flag; never binds)
  head_on      any road that is not motorway-like (a flag; never binds)
  motorway     only when ALL of: divided in >= divided_share of the images;
               separated (barrier or wide median; the divided share stands in
               where median construction is not measured) in >= separation_share;
               nothing vulnerable or a junction seen in any image; and at least
               `absence_images` images that a detector actually ran on.

Rules and speeds: weights/vision/vision_config.json ("rung").
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, name: str, fill: float = 0.0) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), fill)
    return pd.to_numeric(df[name], errors="coerce").fillna(fill).to_numpy(dtype="float64")


def vision_rung(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-segment rung from the vision columns. NaN / 'no imagery' without images."""
    rc = cfg["rung"]
    r, order, thr = rc["rules"], rc["order"], rc["thresholds"]
    n = _col(df, "v_n_forward")
    has = n >= 1
    ped_share = np.maximum.reduce([_col(df, c) for c in
                                   ("v_ped_seen", "v_sidewalk", "v_crossing", "v_school")])
    ped = (ped_share >= r["ped_share"]) | (_col(df, "v_urban") >= r["urban_share"])
    cycle = _col(df, "v_bicycle_seen") >= r["cycle_share"]
    seen_any = np.maximum.reduce([_col(df, c) for c in (
        "v_ped_seen_any", "v_sidewalk_any", "v_crossing_any", "v_moto_seen_any",
        "v_bicycle_seen_any", "v_intersection_any", "v_school_any")])
    barrier = _col(df, "v_median_barrier", np.nan)
    wide = _col(df, "v_median_wide", np.nan)
    divided = _col(df, "v_divided")
    separated = np.where(np.isnan(barrier) & np.isnan(wide), divided,
                         np.nan_to_num(barrier) + np.nan_to_num(wide))
    n_ev = _col(df, "v_n_evidence", np.nan)
    n_ev = np.where(np.isnan(n_ev), n, n_ev)
    motorway_like = ((divided >= r["divided_share"])
                     & (separated >= r["separation_share"])
                     & (seen_any == 0) & (n_ev >= rc["absence_images"]))
    side = _col(df, "v_intersection_any") > 0
    head_on_flag = _col(df, "v_undivided") >= r["head_on_share"]

    applies = {
        "vru_ped": ped,
        "vru_cycle": cycle,
        "vru_ptw": ~motorway_like,
        "side_impact": side & ~motorway_like,
        "head_on": ~motorway_like,
        "motorway": np.ones(len(df), dtype=bool),
    }
    speeds = {k: thr[k]["speed"] for k in order}
    key = np.full(len(df), "motorway", dtype=object)
    best = np.full(len(df), np.inf)
    for k in order:
        better = applies[k] & (speeds[k] < best)
        best = np.where(better, speeds[k], best)
        key = np.where(better, k, key)

    out = pd.DataFrame(index=df.index)
    out["vis_safe_speed"] = np.where(has, best, np.nan)
    out["vis_crash_type_key"] = np.where(has, key, None)
    out["vis_safe_speed_reason"] = [thr[k]["title"] if h else None for k, h in zip(key, has)]
    out["vis_motorway_like"] = np.where(has, motorway_like, None)
    out["vis_conflict_head_on"] = np.where(has, head_on_flag, None)
    out["vis_conflict_side"] = np.where(has, side, None)
    out["vis_ped_evidence"] = np.where(has, np.round(ped_share, 3), np.nan)
    out["vis_status"] = np.where(has, "imagery", "no imagery")
    return out
