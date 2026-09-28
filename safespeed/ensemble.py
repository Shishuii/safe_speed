"""Research ensemble: operating speed (v85) from road context, the road graph and street images.

    from safespeed import ensemble
    nodes, edges = ensemble.read_graph("data/graph_sample")
    ens = ensemble.load()                       # default stack: 7 members
    pred = ens.predict(nodes, edges)            # one row per node: member and stacked log v85
    served = ens.served(nodes, pred)           # one row per served segment (old_id)

Eight members are shipped. Each predicts log(v85) for each road piece (a short
TomTom piece of a SafeSpeed segment; a "node" of the road graph):

  * 4 gradient-boosted tree models (gbm_*), evaluated with the same
    `TreeEnsemble` as the design-speed model; gbm_c2g_vision_irap also takes
    the DINOv3 ThaiRAP classifier's a_* scores;
  * 4 GraphSAGE networks (gnn_*) on two road graphs: `topological` (segments that
    share an end point) and `knn` (city2graph's 12 nearest segment midpoints).

A non-negative linear stack combines them; `exp` gives km/h. A served road
segment (`old_id`) is the (sample_size + 1)-weighted mean of its children's km/h.
Stacks (stack.json): `default` (7 members: all but the DINOv3 classifier member
gbm_c2g_vision_irap, whose small gain is matched by a placebo), `with_dinov3`
(all 8) and `without_images` (the 4 members with no image features). Only the
chosen stack's members are loaded and run.

Inputs are checked before anything is predicted (inputs.json: the countries, the
latitude and longitude range of each network, and the columns that are never
blank in training). Every problem found is listed in one GraphInputError.

Only numpy and pandas are used. The GraphSAGE layers are re-stated in numpy:
per layer, out_i = W_l . mean_{j -> i}(x_j) + b_l + W_r . x_i (the mean counts
duplicate and self-loop edges and is 0 for a node with no incoming edge), then
ReLU; a skip Linear on the standardised input; a head Linear-ReLU-Linear.

A GNN output depends on the node's 2-hop in-neighbourhood. On a sub-graph, only
nodes whose whole 2-hop in-neighbourhood is present (`is_target == 1` in the
sample) get the full-graph prediction; the others are computed but not valid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .design_speed import TreeEnsemble

ENSEMBLE_DIR = Path(__file__).resolve().parent.parent / "weights" / "ensemble"
_ID_COLS = {"tt_id": str, "old_id": str, "country": str}
ID_COLUMNS = ("tt_id", "old_id", "country", "sample_size")      # required besides the features
_CHUNK = 100_000          # edges per aggregation chunk (bounds memory on large graphs)
DEFAULT_STACK = "default"                       # the top-level stack of stack.json
STACKS = (DEFAULT_STACK, "with_dinov3", "without_images")


class GraphInputError(ValueError):
    """The graph inputs cannot be used; the message lists every problem."""


# ------------------------------------------------------------------ readers
def read_nodes(path) -> pd.DataFrame:
    """Node table at full precision (ids as text), as one consolidated frame."""
    return pd.read_csv(path, float_precision="round_trip", dtype=_ID_COLS).copy()


def _edge_problems(e: pd.DataFrame, name: str) -> list:
    missing = [c for c in ("src", "dst") if c not in e.columns]
    if missing:
        return [f"{name} lacks the column(s) {', '.join(missing)}"]
    out = []
    for c in ("src", "dst"):
        v = pd.to_numeric(e[c], errors="coerce")
        bad = v.isna() | ~np.isfinite(v) | (v != np.round(v))
        if bad.any():
            rows = ", ".join(str(i + 2) for i in np.flatnonzero(bad.to_numpy())[:5])
            out.append(f"{name}: {c} must be a whole row number of nodes.csv; not so on "
                       f"{int(bad.sum())} line(s) (file line {rows}"
                       + (" and more)" if bad.sum() > 5 else ")"))
    return out


def read_edges(path) -> tuple[np.ndarray, np.ndarray]:
    """(src, dst) row positions into the node table: messages flow src -> dst."""
    e = pd.read_csv(path)
    problems = _edge_problems(e, Path(path).name)
    if problems:
        raise GraphInputError("the graph cannot be used:\n  - " + "\n  - ".join(problems))
    return (pd.to_numeric(e["src"]).to_numpy(np.int64),
            pd.to_numeric(e["dst"]).to_numpy(np.int64))


def read_graph(folder) -> tuple[pd.DataFrame, dict]:
    """nodes.csv, edges_topological.csv and edges_knn.csv from one folder."""
    d = Path(folder)
    nodes = read_nodes(d / "nodes.csv")
    edges, problems = {}, []
    for g in ("topological", "knn"):
        try:
            edges[g] = read_edges(d / f"edges_{g}.csv")
        except GraphInputError as err:
            problems += str(err).split("\n  - ")[1:]
    if problems:
        raise GraphInputError("the graph cannot be used:\n  - " + "\n  - ".join(problems))
    return nodes, edges


# ------------------------------------------------------------------ GraphSAGE
def mean_neighbours(x: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """mean over incoming edges j -> i of x_j (duplicates counted; 0 if none), float32."""
    n, d = x.shape
    acc = np.zeros((n, d), dtype=np.float64)
    if len(src):
        order = np.argsort(dst, kind="stable")
        s_sorted, d_sorted = src[order], dst[order]
        for a in range(0, len(order), _CHUNK):
            ds = d_sorted[a:a + _CHUNK]
            start = np.flatnonzero(np.r_[True, ds[1:] != ds[:-1]])
            acc[ds[start]] += np.add.reduceat(x[s_sorted[a:a + _CHUNK]].astype(np.float64),
                                              start, axis=0)
    cnt = np.bincount(dst, minlength=n).astype(np.float64)
    return (acc / np.maximum(cnt, 1.0)[:, None]).astype(np.float32)


def sage_layer(x, src, dst, Wl, bl, Wr) -> np.ndarray:
    """torch_geometric SAGEConv(aggr='mean', root_weight=True, normalize=False)."""
    return mean_neighbours(x, src, dst) @ Wl.T + bl + x @ Wr.T


class GraphSAGE:
    """One exported GNN member: frozen standardisation + numpy forward pass."""

    def __init__(self, meta: dict, weights: dict):
        self.meta = meta
        self.graph = meta["graph"]
        self.features = list(meta["features"])
        st = meta["standardise"]
        self.median = np.array([st[c]["median"] for c in self.features], dtype=np.float64)
        self.mean = np.array([st[c]["mean"] for c in self.features], dtype=np.float64)
        self.sd = np.array([st[c]["sd"] for c in self.features], dtype=np.float64)
        self.W = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}

    def standardise(self, X: np.ndarray) -> np.ndarray:
        """NaN -> median; z = (x - mean) / sd (0 where sd is 0); float64, then float32."""
        X = np.asarray(X, dtype=np.float64)
        v = np.where(np.isnan(X), self.median, X)
        ok = np.isfinite(self.sd) & (self.sd != 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.where(ok, (v - self.mean) / np.where(ok, self.sd, 1.0), 0.0)
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)

    def forward(self, x: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        W = self.W
        h = np.maximum(sage_layer(x, src, dst, W["c1.lin_l.weight"], W["c1.lin_l.bias"],
                                  W["c1.lin_r.weight"]), 0)
        h = np.maximum(sage_layer(h, src, dst, W["c2.lin_l.weight"], W["c2.lin_l.bias"],
                                  W["c2.lin_r.weight"]), 0)
        h = h + x @ W["skip.weight"].T + W["skip.bias"]
        h = np.maximum(h @ W["head.0.weight"].T + W["head.0.bias"], 0)
        return (h @ W["head.2.weight"].T + W["head.2.bias"])[:, 0]

    def predict(self, X: np.ndarray, edges: tuple) -> np.ndarray:
        """log v85 per node (float64 of the float32 network output)."""
        src, dst = edges
        return self.forward(self.standardise(X), src, dst).astype(np.float64)


class GBM:
    """One exported tree member (raw features, NaN allowed)."""

    def __init__(self, spec: dict):
        self.features = list(spec["features"])
        self.trees = TreeEnsemble(spec)

    def predict(self, X: np.ndarray, edges=None) -> np.ndarray:
        return self.trees.predict(X)


# ------------------------------------------------------------------ ensemble
class Ensemble:
    """The members of one stack plus its weights. `stack`: 'default' (7 members, no DINOv3
    classifier member), 'with_dinov3' (8) or 'without_images' (4)."""

    def __init__(self, weights_dir: Path = ENSEMBLE_DIR, stack: str = DEFAULT_STACK):
        wd = Path(weights_dir)
        spec = json.loads((wd / "stack.json").read_text())
        if stack == DEFAULT_STACK:
            self.stack_name, st = DEFAULT_STACK, spec
        else:
            alts = spec.get("alternatives", {})
            if stack not in alts:
                raise ValueError(f"unknown stack {stack!r}; choose {DEFAULT_STACK!r} or one of "
                                 f"{sorted(alts)}")
            self.stack_name, st = stack, alts[stack]
        self.inputs = json.loads((wd / "inputs.json").read_text())
        self.stack_members = list(st["members"])
        self.stack_weights = {m: float(st["weights"][m]) for m in self.stack_members}
        self.intercept = float(st["intercept"])
        self.members = {}
        for m in self.stack_members:
            meta = json.loads((wd / f"{m}.json").read_text())
            if m.startswith("gbm"):
                self.members[m] = GBM(meta)
            else:
                with np.load(wd / f"{m}.npz", allow_pickle=False) as z:
                    self.members[m] = GraphSAGE(meta, {k: z[k] for k in z.files})

    def feature_columns(self) -> list:
        out = []
        for mem in self.members.values():
            out += [c for c in mem.features if c not in out]
        return out

    def check_nodes(self, nodes: pd.DataFrame, edges: dict) -> pd.DataFrame:
        """Check the node table and the edge lists against inputs.json. Returns the
        numeric columns (float64); raises GraphInputError listing every problem."""
        rules = self.inputs
        if len(nodes) == 0:
            raise GraphInputError("the graph cannot be used:\n  - nodes.csv has no rows")
        problems = []
        feats = self.feature_columns()
        missing = [c for c in (*ID_COLUMNS, *feats) if c not in nodes.columns]
        if missing:
            problems.append(f"nodes.csv lacks {len(missing)} column(s): {', '.join(missing)} "
                            "(see COLUMNS.md and weights/ensemble/MODEL_CARD.md)")
        key = (nodes["tt_id"].astype(str) if "tt_id" in nodes.columns
               else pd.Series([f"row {i + 1}" for i in range(len(nodes))], index=nodes.index))

        def where(mask) -> str:
            ids = key[mask].tolist()
            return ", ".join(ids[:5]) + (f" and {len(ids) - 5} more" if len(ids) > 5 else "")

        def blank(s: pd.Series) -> pd.Series:
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                return s.isna()
            return s.isna() | s.astype(str).str.strip().eq("")

        if "tt_id" in nodes.columns and blank(nodes["tt_id"]).any():
            problems.append(f"tt_id is blank in {int(blank(nodes['tt_id']).sum())} row(s) "
                            f"({where(blank(nodes['tt_id']))})")
        ctry = None
        if "country" in nodes.columns:
            ctry = nodes["country"].astype(str).str.strip().str.upper().where(~blank(nodes["country"]))
            bad = ~ctry.isin(rules["countries"])
            if bad.any():
                seen = sorted({"(blank)" if pd.isna(v) else v for v in ctry[bad]})
                problems.append(f"country must be one of {', '.join(rules['countries'])}; got "
                                f"{', '.join(seen)} ({where(bad)}). The models are fitted to "
                                "these two networks only.")
        num = {}
        for c in [c for c in ("sample_size", "is_target", *feats) if c in nodes.columns]:
            raw = nodes[c]
            v = pd.to_numeric(raw, errors="coerce").astype("float64")
            text = ~blank(raw) & v.isna()
            if text.any():
                problems.append(f"{c} has values that are not numbers, e.g. "
                                f"'{raw[text].iloc[0]}' ({where(text)})")
            if np.isinf(v).any():
                problems.append(f"{c} has infinite values ({where(np.isinf(v))})")
            if c in rules["never_blank"] and (v.isna() & ~text).any():
                m = v.isna() & ~text
                problems.append(f"{c} is blank in {int(m.sum())} row(s) ({where(m)}); it is "
                                "never blank in the training data")
            num[c] = v
        if "sample_size" in num and (num["sample_size"] < 0).any():
            problems.append(f"sample_size has negative values ({where(num['sample_size'] < 0)})")
        if "is_target" in num:
            bad = ~num["is_target"].isin([0, 1])
            if bad.any():
                problems.append(f"is_target must be 0 or 1 ({where(bad)})")
        if ctry is not None:
            ok = ctry.isin(rules["countries"])
            if "country_TH" in num:
                bad = ok & num["country_TH"].notna() & (num["country_TH"] != (ctry == "TH"))
                if bad.any():
                    problems.append(f"country_TH does not match country ({where(bad)}); it is "
                                    "1 for TH and 0 for MH")
            if "lat" in num and "lon" in num:
                for c, box in rules["coverage"].items():
                    sel = (ctry == c) & num["lat"].notna() & num["lon"].notna()
                    out = sel & ~(num["lat"].between(*box["lat"]) & num["lon"].between(*box["lon"]))
                    if out.any():
                        problems.append(
                            f"lat/lon lie outside the {c} road network ({where(out)}); expected "
                            f"lat {box['lat'][0]:.1f} to {box['lat'][1]:.1f} and lon "
                            f"{box['lon'][0]:.1f} to {box['lon'][1]:.1f}. Check that lat and lon "
                            "are not swapped.")
        for g in sorted({m.graph for m in self.members.values() if isinstance(m, GraphSAGE)}):
            if g not in edges:
                problems.append(f"the {g} edge list is needed by a graph member")
                continue
            s, d = edges[g]
            if len(s) and (min(s.min(), d.min()) < 0 or max(s.max(), d.max()) >= len(nodes)):
                problems.append(f"edges_{g}.csv points to rows outside nodes.csv (0 to "
                                f"{len(nodes) - 1})")
        if problems:
            raise GraphInputError("the graph cannot be used:\n  - " + "\n  - ".join(problems))
        return pd.DataFrame(num, index=nodes.index)

    def predict(self, nodes: pd.DataFrame, edges: dict) -> pd.DataFrame:
        """Per node: pred_<member> (log v85), contrib_<member> = weight * pred,
        pred_stack (log v85) and v85_pred_kmh."""
        num = self.check_nodes(nodes, edges)
        keep = [c for c in ("idx", "tt_id", "old_id", "country", "sample_size", "is_target")
                if c in nodes.columns]
        out = nodes[keep].copy()
        stack = np.full(len(nodes), self.intercept)
        for m, mem in self.members.items():
            X = num[mem.features].to_numpy(dtype=np.float64)
            p = mem.predict(X, edges.get(getattr(mem, "graph", None)))
            out[f"pred_{m}"] = p
            out[f"contrib_{m}"] = self.stack_weights[m] * p
            stack = stack + self.stack_weights[m] * p
        out["pred_stack"] = stack
        out["v85_pred_kmh"] = np.exp(stack)
        return out

    def served(self, nodes: pd.DataFrame, pred: pd.DataFrame, only_targets: bool = True,
               measured: str | None = "v85_measured") -> pd.DataFrame:
        """Per served segment (old_id): (sample_size + 1)-weighted mean over its children of
        exp(pred_stack) and of each member's exp(pred_<member>)."""
        d = pred.copy()
        if only_targets and "is_target" in d.columns:
            d = d[d["is_target"].astype(int) == 1]
        w = pd.to_numeric(d["sample_size"], errors="coerce").fillna(0).clip(lower=0) + 1.0
        rows = []
        for sid, g in d.groupby("old_id", sort=False):
            wg = w.loc[g.index].to_numpy()
            r = {"old_id": sid, "n_children": int(len(g)),
                 "v85_pred_kmh": float((g["v85_pred_kmh"].to_numpy() * wg).sum() / wg.sum())}
            for m in self.members:
                r[f"v85_{m}_kmh"] = float((np.exp(g[f"pred_{m}"].to_numpy()) * wg).sum() / wg.sum())
            if measured and measured in nodes.columns:
                y = pd.to_numeric(nodes.loc[g.index, measured], errors="coerce").to_numpy()
                ok = np.isfinite(y)
                r["v85_measured"] = float((y[ok] * wg[ok]).sum() / wg[ok].sum()) if ok.any() else np.nan
            rows.append(r)
        return pd.DataFrame(rows)


_cache: dict = {}


def load(weights_dir: Path = ENSEMBLE_DIR, stack: str = DEFAULT_STACK) -> Ensemble:
    key = (str(weights_dir), stack)
    if key not in _cache:
        _cache[key] = Ensemble(weights_dir, stack)
    return _cache[key]
