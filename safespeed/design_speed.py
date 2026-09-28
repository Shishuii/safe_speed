"""Design-speed model: the operating speed a road's layout and surroundings invite.

Four gradient-boosted tree ensembles predict the log of a road's median (v50) and
85th-percentile (v85) speed, one pair per country, from context only: the posted
limit, length, nearby activity, population, geometry and location. They were
trained on measured TomTom speeds with scikit-learn and exported as plain JSON
trees (weights/design_speed_trees.json). This module evaluates them with numpy,
adding the trees in training order, so the predictions are identical to
scikit-learn's.

The model is "grouped-additive": factor groups (location, geometry, activity,
population, posted limit, length) never interact, so on the model's log scale each
group's contribution is exact arithmetic, not an approximation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class TreeEnsemble:
    """A sum of regression trees stored as flat node arrays."""

    def __init__(self, spec: dict):
        self.baseline = float(spec["baseline"])
        feat, thr, miss, left, right, leaf, value, roots = [], [], [], [], [], [], [], []
        offset = 0
        for t in spec["trees"]:
            n = len(t["value"])
            roots.append(offset)
            feat += t["feature"]
            thr += [float(x) for x in t["threshold"]]          # "inf" -> inf
            miss += t["missing_left"]
            left += [c + offset for c in t["left"]]
            right += [c + offset for c in t["right"]]
            leaf += t["leaf"]
            value += t["value"]
            offset += n
        self.feature = np.asarray(feat, dtype=np.int64)
        self.threshold = np.asarray(thr, dtype=np.float64)
        self.missing_left = np.asarray(miss, dtype=bool)
        self.left = np.asarray(left, dtype=np.int64)
        self.right = np.asarray(right, dtype=np.int64)
        self.leaf = np.asarray(leaf, dtype=bool)
        self.value = np.asarray(value, dtype=np.float64)
        self.roots = np.asarray(roots, dtype=np.int64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Raw prediction (log km/h) for each row of X."""
        X = np.asarray(X, dtype=np.float64)
        rows = np.arange(len(X))
        acc = np.full(len(X), self.baseline)
        for root in self.roots:
            node = np.full(len(X), root)
            while True:
                inner = ~self.leaf[node]
                if not inner.any():
                    break
                x = X[rows, self.feature[node]]
                go_left = np.where(np.isnan(x), self.missing_left[node],
                                   x <= self.threshold[node])
                node = np.where(inner, np.where(go_left, self.left[node],
                                                self.right[node]), node)
            acc += self.value[node]
        return acc


class DesignSpeedModel:
    """Median and 85th-percentile design speed, with exact factor attribution."""

    def __init__(self, path: Path):
        spec = json.loads(Path(path).read_text())
        self.features = spec["features"]
        self.groups = {g: (v["index"], v["label"]) for g, v in spec["factor_groups"].items()}
        self.reference_row = {c: np.asarray(v, dtype=np.float64)
                              for c, v in spec["reference_row"].items()}
        self.models = {tgt: {c: TreeEnsemble(m) for c, m in by_c.items()}
                       for tgt, by_c in spec["models"].items()}
        self.countries = sorted(self.reference_row)
        self.coverage = spec["coverage"]        # lat/lon box of each country's training roads

    # ------------------------------------------------------------ features
    def design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        def col(name):
            return pd.to_numeric(df[name], errors="coerce")

        X = pd.DataFrame(index=df.index)
        X["posted_limit"] = col("posted_limit")
        X["length_log"] = np.log1p(col("length_km").clip(lower=0) * 1000.0)
        X["poi_vru_log"] = np.log1p(col("poi_vru_raw").clip(lower=0))
        X["school_log"] = np.log1p(col("poi_school").clip(lower=0))
        X["transport_log"] = np.log1p(col("poi_transport").clip(lower=0))
        X["pop_log"] = np.log1p(col("pop_exposure").clip(lower=0))
        X["pop_density_log"] = np.log1p(col("pop_density").clip(lower=0))
        for c in ("geo_sinuosity", "geo_curvature", "geo_junction_density", "lat", "lon"):
            X[c] = col(c)
        X["country_TH"] = (df["country"].astype(str) == "TH").astype(float)
        return X[self.features].to_numpy(dtype="float64")

    # ------------------------------------------------------------ predict
    def predict_kmh(self, X: np.ndarray, country: np.ndarray, target: str) -> np.ndarray:
        out = np.full(len(X), np.nan)
        for c, mdl in self.models[target].items():
            sel = country == c
            if sel.any():
                out[sel] = mdl.predict(X[sel])
        return np.exp(out)

    def attribution(self, X: np.ndarray, country: np.ndarray, target: str = "v85"):
        """Contribution (km/h) of each factor group to the design speed.

        Reference road: every input at the country's median value. On the log
        scale, a group's contribution is the change in prediction when that group
        is reset to the reference road; because groups do not interact these add up
        exactly to (log prediction - log reference). The km/h difference from the
        reference road is then shared out in proportion to them, so the km/h values
        also add up exactly, but one group's km/h is not the change from resetting
        that group alone. Returns (contributions [n x groups], reference speed km/h).
        """
        ng = len(self.groups)
        contrib = np.zeros((len(X), ng))
        f_log = np.full(len(X), np.nan)
        base_log = np.full(len(X), np.nan)
        for c, mdl in self.models[target].items():
            sel = country == c
            if not sel.any():
                continue
            ref = self.reference_row[c]
            Xc = X[sel]
            fc = mdl.predict(Xc)
            bc = float(mdl.predict(ref.reshape(1, -1))[0])
            for j, (idx, _) in enumerate(self.groups.values()):
                Xr = Xc.copy()
                Xr[:, idx] = ref[idx]
                contrib[sel, j] = fc - mdl.predict(Xr)
            f_log[sel] = fc
            base_log[sel] = bc
        base_kmh = np.exp(base_log)
        total_kmh = np.exp(f_log) - base_kmh
        denom = contrib.sum(axis=1, keepdims=True)
        share = np.divide(contrib, np.where(np.abs(denom) < 1e-12, np.nan, denom))
        return np.nan_to_num(share) * total_kmh[:, None], base_kmh

    def run(self, df: pd.DataFrame):
        """Design speeds and attribution. Returns (table, unrounded v50, unrounded v85)."""
        X = self.design_matrix(df)
        country = df["country"].astype(str).to_numpy()
        v50 = self.predict_kmh(X, country, "v50")
        v85 = self.predict_kmh(X, country, "v85")
        out = pd.DataFrame(index=df.index)
        out["design_speed_v50"] = np.round(v50, 1)
        out["design_speed_v85"] = np.round(v85, 1)
        obs85 = pd.to_numeric(df["operating_speed"], errors="coerce")   # optional input
        out["speed_vs_design_kmh"] = (obs85 - v85).round(1)

        contrib, base_kmh = self.attribution(X, country, "v85")
        out["speed_attr_base_kmh"] = np.round(base_kmh, 1)
        for j, g in enumerate(self.groups):
            out[f"speed_attr_{g}"] = np.round(contrib[:, j], 2)
        order = np.argsort(-np.abs(contrib), axis=1)
        labels = np.array([lab for _, lab in self.groups.values()])
        out["speed_top_factor"] = labels[order[:, 0]]
        out["speed_top_factor_2"] = labels[order[:, 1]]
        return out, v50, v85
