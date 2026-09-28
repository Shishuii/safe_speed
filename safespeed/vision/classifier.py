"""DINOv3 ThaiRAP classifier: iRAP attribute probabilities from street images.

Each image is described by frozen DINOv3 ViT-L/16 features: five pooled views
(the first token, the mean over all patches, and the left, centre and right
thirds of the image) of 1,024 values each, 5,120 in all. One linear softmax head
per ThaiRAP attribute turns them into the probability of each iRAP code. The
heads were trained by the SafeSpeed team on the ThaiRAP survey images and labels,
so they are non-commercial (weights/vision_thairap/LICENSE.txt).

    p_j = softmax(((x - mu) / sd) @ W_j.T + b_j)

for attribute j, where W_j, b_j and its iRAP codes are rows
offsets[j]:offsets[j+1] of W, b and classes. The arithmetic is float64 on the
float16 features, in chunks of 4,096 images, as SafeSpeed's full build computed it; the
probabilities are then stored as float32.

A segment's score is the mean of its images' float32 probabilities over its
(500 m point, image) rows, summed in float32 with compensation in row order.
That is what the pandas groupby mean of SafeSpeed's full build does, so the package reproduces
its values bit for bit, independently of the pandas version.

Output columns are a_<attribute>__<iRAP code>, for the DESIGN-role attributes
only: they are the inputs of the ensemble member gbm_c2g_vision_irap. The file
also holds heads for the reference-role attributes (who uses the road,
junctions, median, area type, crossings, sidewalks, school zones). Those are
never model inputs and are not output unless asked for (roles=("design", "ref")).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 4096


class DinoV3Classifier:
    """The persisted linear heads (dinov3_heads.npz)."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as z:
            self.keys = [str(k) for k in z["keys"]]
            self.items = dict(zip(self.keys, (str(x) for x in z["items"])))
            self.roles = dict(zip(self.keys, (str(x) for x in z["roles"])))
            self.classes = z["classes"].astype(np.int64)
            self.offsets = z["offsets"].astype(np.int64)
            self.W = z["W"]
            self.b = z["b"]
            self.mu = z["mu"]
            self.sd = z["sd"]
            self.meta = {k[5:]: (z[k].tolist() if z[k].ndim else z[k].item())
                         for k in z.files if k.startswith("meta_")}
        self.feature_dim = int(self.W.shape[1])
        self._Wt = self.W.astype(np.float64).T

    def codes(self, key: str) -> list:
        j = self.keys.index(key)
        return self.classes[self.offsets[j]:self.offsets[j + 1]].tolist()

    def attributes(self, roles=("design",)) -> list:
        return [k for k in self.keys if self.roles[k] in roles]

    def columns(self, roles=("design",)) -> list:
        """a_<attribute>__<code> for the attributes whose role is in `roles`."""
        return [f"a_{k}__{int(c)}" for k in self.attributes(roles) for c in self.codes(k)]

    def predict_proba(self, X: np.ndarray) -> list:
        """Per attribute, (n images x n codes) float64 probabilities."""
        off = self.offsets
        out = [[] for _ in range(len(off) - 1)]
        for i in range(0, len(X), CHUNK):
            Z = (np.asarray(X[i:i + CHUNK], np.float32).astype(np.float64) - self.mu) / self.sd
            L = Z @ self._Wt + self.b
            for j in range(len(off) - 1):
                blk = L[:, off[j]:off[j + 1]]
                e = np.exp(blk - blk.max(1, keepdims=True))
                out[j].append(e / e.sum(1, keepdims=True))
        return [np.concatenate(o) if o else np.zeros((0, off[j + 1] - off[j]))
                for j, o in enumerate(out)]

    def image_scores(self, X: np.ndarray, roles=("design",)) -> pd.DataFrame:
        """float32 probabilities, one column per a_* name, one row per row of X."""
        P = self.predict_proba(X)
        keep = set(self.attributes(roles))
        cols = {}
        for j, k in enumerate(self.keys):
            if k in keep:
                for c, code in enumerate(self.codes(k)):
                    cols[f"a_{k}__{int(code)}"] = P[j][:, c].astype(np.float32)
        return pd.DataFrame(cols, columns=self.columns(roles))


def segment_means(seg_ids, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment float32 mean of `values` rows (compensated sum, in row order).

    seg_ids   one segment id per row
    values    float32 (rows x columns)
    Returns (unique segment ids, sorted; float32 means).
    """
    V = np.asarray(values, dtype=np.float32)
    ids = np.asarray(seg_ids).astype(str)
    uniq, g = np.unique(ids, return_inverse=True)
    n_g = len(uniq)
    s = np.zeros((n_g, V.shape[1]), np.float32)
    comp = np.zeros_like(s)
    count = np.bincount(g, minlength=n_g)
    rank = pd.Series(g).groupby(g).cumcount().to_numpy()      # position within its segment
    for r in range(int(rank.max()) + 1 if len(rank) else 0):
        rows = np.flatnonzero(rank == r)                        # at most one row per segment
        gi = g[rows]
        y = V[rows] - comp[gi]
        t = s[gi] + y
        c = (t - s[gi]) - y
        comp[gi] = np.where(np.isnan(c), np.float32(0), c)
        s[gi] = t
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / count.astype(np.float32)[:, None]
    return uniq, mean.astype(np.float32)
