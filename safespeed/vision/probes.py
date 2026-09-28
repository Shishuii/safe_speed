"""Linear probes on CLIP image embeddings, evaluated from exported weights.

Every probe is a scikit-learn LogisticRegression (lbfgs, C = 1, balanced class
weights) on the 768-d CLIP ViT-L/14 image embedding. This module repeats
scikit-learn's own prediction arithmetic with numpy:

  * two classes:   p = 1 / (1 + exp(-(x . w + b)))            -> [1 - p, p]
  * more classes:  softmax(x . W^T + b), max subtracted first

Probes and what they output per image:

  ThaiRAP probes (weights/vision_thairap/, CC BY-NC 4.0)
      11 iRAP attributes (lighting, shoulders, roadside hazards, grip, calming,
      property access, service road); output = the iRAP code with the highest
      probability.
  OSM probes (weights/vision/osm_probes.npz)
      divided  P(the road is a divided carriageway); v_divided = P >= 0.5
      lanes    expected lane count, sum over 1-4 of P(k) * k
      Five cross-fitted models per probe (50 km spatial folds of the Thai
      network): an image of the training network uses the model of its fold,
      which never saw that segment's OSM label. That is how the pipeline made
      its values, so that is what reproduces them. An image from anywhere else
      (osm_fold blank) uses the extra model refitted on all labelled images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _proba(X: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    scores = X @ coef.T + intercept                  # float32 X is promoted to float64
    if coef.shape[0] == 1:                           # binary: logistic of one score
        p = 1.0 / (1.0 + np.exp(-scores.reshape(-1)))
        return np.vstack([1 - p, p]).T
    scores = scores - scores.max(axis=1).reshape(-1, 1)
    np.exp(scores, out=scores)
    scores /= scores.sum(axis=1).reshape(-1, 1)
    return scores


class Probe:
    """One logistic-regression probe."""

    def __init__(self, coef, intercept, classes, name: str = ""):
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        self.classes = np.asarray(classes)
        self.name = name

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _proba(X, self.coef, self.intercept)

    def predict_code(self, X: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.predict_proba(X), axis=1)].astype(int)


def _probe(z, prefix: str, name: str) -> Probe:
    return Probe(z[f"{prefix}coef"], z[f"{prefix}intercept"], z[f"{prefix}classes"], name)


class ThaiRAPProbes:
    """The 11 ThaiRAP attribute probes (non-commercial licence, see their folder)."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.attributes = [str(a) for a in z["attributes"]]
        self.items = dict(zip(self.attributes, (str(x) for x in z["irap_items"])))
        self.probes = {a: _probe(z, f"{a}__", a) for a in self.attributes}
        self.class_names = {a: dict(zip(z[f"{a}__classes"].astype(int).tolist(),
                                        (str(x) for x in z[f"{a}__class_names"])))
                            for a in self.attributes}

    def codes(self, X: np.ndarray) -> dict:
        """iRAP code per image and attribute."""
        return {a: p.predict_code(X) for a, p in self.probes.items()}


class OSMProbes:
    """Divided / lane-count probes: five cross-fitted fold models + an all-data refit."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.folds = [int(k) for k in z["folds"]]
        self.models = {}
        for kind in ("divided", "lanes"):
            self.models[kind] = {k: _probe(z, f"{kind}__fold{k}__", f"{kind} fold {k}")
                                 for k in self.folds if f"{kind}__fold{k}__coef" in z.files}
            self.models[kind]["all"] = _probe(z, f"{kind}__all__", f"{kind} all-data refit")

    @staticmethod
    def _value(p: Probe, X: np.ndarray, kind: str) -> np.ndarray:
        P = p.predict_proba(X)
        if kind == "divided":
            return P[:, list(p.classes).index(1)]
        return P @ p.classes

    def predict(self, X: np.ndarray, fold: np.ndarray, kind: str) -> np.ndarray:
        """P(divided) or expected lanes per image. `fold` is the image's OSM fold
        (0-4) or -1 for an image outside the training network (all-data refit).
        A fold whose model could not be trained gives NaN, as in the pipeline."""
        out = np.full(len(X), np.nan)
        for k in np.unique(fold):
            te = fold == k
            key = "all" if k < 0 else int(k)
            if key in self.models[kind]:
                out[te] = self._value(self.models[kind][key], X[te], kind)
        return out
