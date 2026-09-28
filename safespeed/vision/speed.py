"""Vision-only speed model: the speed the road in the photos invites.

Two gradient-boosted tree ensembles (scikit-learn HistGradientBoostingRegressor,
1,000 depth-2 trees each) predict log v50 and log v85 from 56 image features:
the 24 iRAP design columns (13 of them are never measured by the image route and
are always blank) and the 32 embedding components. Thailand only. Exported to
weights/vision/vision_speed_trees.json and evaluated with the package's
TreeEnsemble, adding the trees in training order, so the predictions equal
scikit-learn's.

Attribution is exact at factor-group level: the groups never interact inside a
tree (grouped-additive), so on the log scale the change from resetting a group
to the reference row (the training median of each feature) is exact arithmetic.
The km/h difference from the reference road is shared out in proportion to
those log contributions.

Expected harm (`vis_harm`) = the Lubbe, Wu & Jeppsson (2022) injury risk of the
vision rung's road user, averaged over the normal speed distribution pinned by
the predicted v50 and v85 (safespeed/harm.py, weights/injury_curves.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..design_speed import TreeEnsemble
from ..harm import harm_from_v50_v85


class VisionSpeedModel:
    def __init__(self, path: Path):
        spec = json.loads(Path(path).read_text())
        self.features = spec["features"]
        self.country = spec["country"]
        self.groups = {g: (v["index"], v["label"]) for g, v in spec["factor_groups"].items()}
        self.ref = np.asarray(spec["reference_row"][self.country], dtype=np.float64)  # null -> NaN
        self.models = {t: TreeEnsemble(by_c[self.country]) for t, by_c in spec["models"].items()}
        self.min_images = int(spec.get("min_forward_images", 1))

    def usable(self, seg: pd.DataFrame) -> np.ndarray:
        n = pd.to_numeric(seg.get("v_n_forward"), errors="coerce") if "v_n_forward" in seg \
            else pd.Series(0, index=seg.index)
        return (seg["country"].astype(str).eq(self.country)
                & n.fillna(0).ge(self.min_images)).to_numpy()

    def design_matrix(self, seg: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(index=seg.index)
        for c in self.features:
            X[c] = pd.to_numeric(seg[c], errors="coerce") if c in seg.columns else np.nan
        return X.to_numpy(dtype="float64")

    def decompose(self, X: np.ndarray):
        """(contributions km/h [n x groups], reference speed km/h)."""
        mdl = self.models["v85"]
        f = mdl.predict(X)
        b = float(mdl.predict(self.ref.reshape(1, -1))[0])
        contrib = np.zeros((len(X), len(self.groups)))
        for j, (idx, _) in enumerate(self.groups.values()):
            Xr = X.copy()
            Xr[:, idx] = self.ref[idx]
            contrib[:, j] = f - mdl.predict(Xr)
        base = np.exp(b)
        total = np.exp(f) - base
        denom = contrib.sum(axis=1, keepdims=True)
        share = np.divide(contrib, np.where(np.abs(denom) < 1e-12, np.nan, denom))
        return np.nan_to_num(share) * total[:, None], base

    def run(self, seg: pd.DataFrame, crash_keys, curves: dict) -> pd.DataFrame:
        out = pd.DataFrame(index=seg.index)
        cols = (["vis_pred_v50", "vis_pred_v85", "vis_harm"]
                + [f"vis_speed_attr_{g}" for g in self.groups]
                + ["vis_speed_attr_base_kmh", "vis_speed_top_factor"])
        for c in cols:
            out[c] = np.nan if c != "vis_speed_top_factor" else None
        use = self.usable(seg)
        if not use.any():
            return out
        X = self.design_matrix(seg.loc[use])
        v50 = np.exp(self.models["v50"].predict(X))
        v85 = np.exp(self.models["v85"].predict(X))
        keys = np.asarray(crash_keys, dtype=object)[use]
        out.loc[use, "vis_pred_v50"] = np.round(v50, 1)
        out.loc[use, "vis_pred_v85"] = np.round(v85, 1)
        out.loc[use, "vis_harm"] = harm_from_v50_v85(curves, v50, v85, keys)
        contrib, base = self.decompose(X)
        for j, g in enumerate(self.groups):
            out.loc[use, f"vis_speed_attr_{g}"] = np.round(contrib[:, j], 2)
        out.loc[use, "vis_speed_attr_base_kmh"] = round(float(base), 1)
        labels = np.array([lab for _, lab in self.groups.values()])
        out.loc[use, "vis_speed_top_factor"] = labels[np.argmax(np.abs(contrib), axis=1)]
        return out
