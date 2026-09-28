"""SafeSpeed design-speed model: the speed a road's layout and surroundings invite.

    import safespeed
    pred = safespeed.predict(safespeed.read_csv("data/sample_segments.csv"))
    safespeed.write_csv(pred, "my_predictions.csv")

`predict` takes one row per road segment (columns in INPUT_COLUMNS) and returns
the predicted median (v50) and 85th-percentile (v85) speed, the exact
contribution of each factor group to v85, and the expected harm those speeds
imply. Everything it needs is in the `weights/` folder next to this package.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .design_speed import DesignSpeedModel
from .harm import harm_from_v50_v85

__version__ = "1.0"
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

#: column -> (required, blank allowed, meaning). Required columns must be present.
#: A blank is only accepted where "blank allowed" is True; optional columns may be
#: left out entirely.
INPUT_COLUMNS = {
    "id":                   (True,  False, "segment identifier"),
    "country":              (True,  False, "MH (Maharashtra) or TH (Thailand)"),
    "posted_limit":         (True,  True,  "posted speed limit, km/h"),
    "length_km":            (True,  False, "segment length, km"),
    "lat":                  (True,  False, "latitude of the segment's first point"),
    "lon":                  (True,  False, "longitude of the segment's first point"),
    "poi_vru_raw":          (True,  False, "weighted count of places that bring people on foot (see COLUMNS.md)"),
    "poi_school":           (True,  False, "schools nearby"),
    "poi_transport":        (True,  False, "public-transport stops nearby"),
    "pop_exposure":         (True,  False, "residents within 300 m (WorldPop 2025)"),
    "pop_density":          (True,  False, "mean residents per 100 m cell within 300 m"),
    "geo_sinuosity":        (True,  False, "road length / straight-line length"),
    "geo_curvature":        (True,  False, "total turning angle per metre, rad/m"),
    "geo_junction_density": (True,  False, "other network segment ends within 35 m of this segment's ends, per km"),
    "name":                 (False, True,  "road name (passed through)"),
    "operating_speed":      (False, True,  "measured 85th-percentile speed, km/h (for comparison only)"),
    "crash_type":           (False, True,  "who is most at risk, for harm_model: vru_ped, vru_ptw, motorway, ... (blank: vru_ped)"),
}

#: Output columns, in the order they are written.
OUTPUT_COLUMNS = [
    "id", "country", "name",
    "design_speed_v50", "design_speed_v85", "speed_vs_design_kmh",
    "speed_attr_base_kmh", "speed_attr_location", "speed_attr_geometry",
    "speed_attr_activity", "speed_attr_population", "speed_attr_posted_limit",
    "speed_attr_length", "speed_top_factor", "speed_top_factor_2",
    "harm_model",
]


class InputError(ValueError):
    """The input table cannot be used; the message says what to fix."""


_cache: dict = {}


def load(weights_dir: Path = WEIGHTS_DIR):
    """Load (model, injury curves) once."""
    key = str(weights_dir)
    if key not in _cache:
        wd = Path(weights_dir)
        _cache[key] = (DesignSpeedModel(wd / "design_speed_trees.json"),
                       json.loads((wd / "injury_curves.json").read_text()))
    return _cache[key]


def read_csv(path) -> pd.DataFrame:
    """Read an input CSV at full precision.

    Use this rather than a plain pd.read_csv: the trees split on exact input
    values, so rounding an input in the file can move a road's design speed by a
    few km/h.
    """
    return pd.read_csv(path, float_precision="round_trip", dtype={"id": str, "name": str})


#: Decimal places kept when results are written to CSV (write_csv), by column name: a
#: key that starts with "_" matches the end of a name, any other key its start. These
#: columns come from float32 arithmetic (the ensemble's graph networks, the CLIP
#: embedding components v_emb_*, the DINOv3 classifier scores a_*), whose last digits
#: change with the numpy version and the processor. Rounded like this, every platform
#: writes the same file. Other columns keep every digit, and the Python functions
#: return every column unrounded.
CSV_DECIMALS = {"pred_": 4, "contrib_": 4, "_kmh": 2, "v_emb_": 5, "a_": 5}


def _csv_decimals(column) -> int | None:
    name = str(column)
    for key, n in CSV_DECIMALS.items():
        if name.endswith(key) if key.startswith("_") else name.startswith(key):
            return n
    return None


def write_csv(df: pd.DataFrame, path) -> None:
    """Write a results table to CSV, with the float32 columns rounded (CSV_DECIMALS)."""
    out = df.copy()
    for c in out.columns:
        n = _csv_decimals(c)
        if n is not None and pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].astype("float64").round(n) + 0.0     # + 0.0 turns -0.0 into 0.0
    out.to_csv(path, index=False)


def _ids(d: pd.DataFrame, mask) -> str:
    ids = d.loc[mask, "id"].astype(str).tolist()
    return ", ".join(ids[:5]) + (f" and {len(ids) - 5} more" if len(ids) > 5 else "")


def _rows(n: int) -> str:
    return f"{n} row" + ("" if n == 1 else "s")


def prepare(df: pd.DataFrame, countries, coverage: dict, crash_types) -> pd.DataFrame:
    """Check and clean the input table (same cleaning as the SafeSpeed pipeline).

    Raises InputError listing every problem found, so a file can be fixed in one go.
    """
    if len(df) == 0:
        raise InputError("the input has no rows")
    missing = [c for c, (req, _, _) in INPUT_COLUMNS.items() if req and c not in df.columns]
    if missing:
        raise InputError("missing required columns: " + ", ".join(missing)
                         + " (see COLUMNS.md)")
    d = df.copy()
    for c in INPUT_COLUMNS:
        if c not in d.columns:
            d[c] = np.nan
    problems = []

    def blank(v):
        if isinstance(v, str):
            return not v.strip()
        return v is None or bool(pd.isna(v))

    d["id"] = [None if blank(v) else str(v).strip() for v in d["id"]]
    d["country"] = [None if blank(v) else str(v).strip().upper() for v in d["country"]]
    d["name"] = [None if blank(v) else str(v) for v in d["name"]]
    d["crash_type"] = ["vru_ped" if blank(v) else str(v).strip().lower() for v in d["crash_type"]]
    bad = ~d["crash_type"].isin(crash_types)
    if bad.any():
        problems.append(f"crash_type must be one of {', '.join(crash_types)} or blank; got "
                        f"{', '.join(sorted(set(d.loc[bad, 'crash_type'])))} (rows {_ids(d, bad)})")
    if d["id"].isna().any():
        pos = [str(i + 1) for i in np.flatnonzero(d["id"].isna().to_numpy())]
        problems.append(f"id is blank in {_rows(len(pos))} (data row "
                        f"{', '.join(pos[:5])}{' and more' if len(pos) > 5 else ''})")
    bad = d["country"].isna() | ~d["country"].isin(countries)
    if bad.any():
        seen = sorted({"(blank)" if pd.isna(v) else str(v) for v in d.loc[bad, "country"]})
        problems.append(f"country must be one of {', '.join(countries)}; got "
                        f"{', '.join(seen)} (rows {_ids(d, bad)}). The model is fitted "
                        "to these two networks only.")

    for c, (_, blank_ok, _) in INPUT_COLUMNS.items():
        if c in ("id", "country", "name", "crash_type"):
            continue
        raw = d[c]
        num = pd.to_numeric(raw.astype(object).where(raw.notna(), np.nan),
                            errors="coerce").astype("float64")
        text = raw.map(lambda v: isinstance(v, str) and bool(v.strip()))
        unreadable = (text & num.isna()) | np.isinf(num)
        if unreadable.any():
            problems.append(f"{c} has values that are not numbers, e.g. "
                            f"'{raw[unreadable].iloc[0]}' (rows {_ids(d, unreadable)})")
        if not blank_ok and (num.isna() & ~unreadable).any():
            m = num.isna() & ~unreadable
            problems.append(f"{c} is blank in {_rows(int(m.sum()))} ({_ids(d, m)}); "
                            "this column cannot be blank")
        if c not in ("lat", "lon") and (num < 0).any():
            problems.append(f"{c} has negative values (rows {_ids(d, num < 0)})")
        d[c] = num
    # location must lie within the country's training roads (catches swapped lat/lon)
    for c, box in coverage.items():
        sel = (d["country"] == c) & d["lat"].notna() & d["lon"].notna()
        out = sel & ~(d["lat"].between(*box["lat"]) & d["lon"].between(*box["lon"]))
        if out.any():
            problems.append(f"lat/lon lie outside the {c} road network (rows {_ids(d, out)}); "
                            f"expected lat {box['lat'][0]:.1f} to {box['lat'][1]:.1f} and lon "
                            f"{box['lon'][0]:.1f} to {box['lon'][1]:.1f}. Check that lat and "
                            "lon are not swapped.")
    if problems:
        raise InputError("the input cannot be used:\n  - " + "\n  - ".join(problems))

    # A value of 0 is a placeholder in the source data, not a real limit or speed.
    for c in ("posted_limit", "operating_speed"):
        d.loc[d[c] == 0, c] = np.nan
    return d


def predict(df: pd.DataFrame, weights_dir: Path = WEIGHTS_DIR) -> pd.DataFrame:
    """Predict design speeds for road segments. One row per input row, same index."""
    model, curves = load(weights_dir)
    d = prepare(df.reset_index(drop=True), model.countries, model.coverage,
                list(curves["crash_user"]))
    ds, v50, v85 = model.run(d)
    ds["harm_model"] = harm_from_v50_v85(curves, v50, v85, d["crash_type"].to_numpy(dtype=object))
    out = pd.concat([d[["id", "country", "name"]], ds], axis=1)[OUTPUT_COLUMNS]
    out.index = df.index
    return out
