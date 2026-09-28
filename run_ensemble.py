#!/usr/bin/env python3
"""Run the SafeSpeed research ensemble (v85 from context, road graph and street images).

    python3 run_ensemble.py                          # the graph sample in data/graph_sample/,
                                                     # default stack (7 members)
    python3 run_ensemble.py --check                  # ... and compare with its expected.csv
    python3 run_ensemble.py --stack with_dinov3      # 8 members: adds the DINOv3 classifier
                                                     # member (its small gain is matched by
                                                     # a placebo)
    python3 run_ensemble.py --stack without_images   # 4 members: no image features at all
    python3 run_ensemble.py --vision-from-images     # recompute the Thai target pieces' image
                                                     # features (v_*; with --stack with_dinov3
                                                     # also a_*) from the vision sample and
                                                     # check that the two models chain
    python3 run_ensemble.py --graph my_folder/ --output my_ensemble.csv

Writes outputs/ensemble_predictions.csv (one row per target road piece). Needs numpy
and pandas only. A graph with bad rows is refused with a list of every problem (exit
code 2). See weights/ensemble/MODEL_CARD.md.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True          # keep the repository free of __pycache__

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "data" / "graph_sample"
VSAMPLE = HERE / "data" / "vision_sample"
sys.path.insert(0, str(HERE))
from safespeed import ensemble, write_csv  # noqa: E402

TOL = 1e-5            # relative, for GNN members, the stack and km/h (float32 networks)
EMB_TOL = 1e-6        # v_emb_* from the image route are float32 PCA scores (see run_vision.py)
A_TOL = 1e-6          # a_* are float32 classifier means (bit-identical on the reference machine)


def _short(m: str) -> str:
    return (m.replace("gbm_", "G:").replace("gnn_topological", "N:topo")
             .replace("gnn_c2g_knn", "N:knn").replace("_vision", "+v").replace("_irap", "+d"))


def show(served: pd.DataFrame, ens: ensemble.Ensemble) -> None:
    mem = list(ens.members)
    print(f"\nstack '{ens.stack_name}': log v85 = {ens.intercept:.4f} + "
          + " + ".join(f"{ens.stack_weights[m]:.3f}*{_short(m)}" for m in mem))
    head = f"{'served id':<10} {'kids':>4} {'v85 pred':>8} {'meas.':>6}  " + " ".join(
        f"{_short(m):>9}" for m in mem)
    print("\n" + head)
    for r in served.itertuples(index=False):
        d = r._asdict()
        meas = d.get("v85_measured", np.nan)
        print(f"{d['old_id']:<10} {d['n_children']:>4} {d['v85_pred_kmh']:>8.1f} "
              f"{'' if not np.isfinite(meas) else f'{meas:.1f}':>6}  "
              + " ".join(f"{d[f'v85_{m}_kmh']:>9.1f}" for m in mem))
    dino = any(m.endswith("_irap") for m in mem)
    print("\nkm/h. v85 pred = stacked prediction; meas. = measured TomTom v85; member columns = "
          "each member's own v85\n(G = trees, N = GraphSAGE on the topological or KNN graph, "
          "+v = with street-image features"
          + (", +d = with the DINOv3\nThaiRAP classifier's scores); " if dino else "); ")
          + "all are (sample_size + 1)-weighted means over the segment's road pieces.")


def expected_columns(ens) -> tuple[str, str]:
    """expected.csv columns of the chosen stack (log v85, km/h)."""
    if ens.stack_name == ensemble.DEFAULT_STACK:
        return "pred_stack", "v85_pred_kmh"
    return f"pred_stack_{ens.stack_name}", f"v85_pred_kmh_{ens.stack_name}"


def rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))) if len(a) else 0.0


def check(pred: pd.DataFrame, served: pd.DataFrame, exp: pd.DataFrame, ens) -> bool:
    en = exp[exp["level"] == "node"].set_index("tt_id")
    es = exp[exp["level"] == "served"].set_index("old_id")
    p = pred[pred["is_target"] == 1].set_index("tt_id")
    ok = set(p.index) == set(en.index)
    if not ok:
        print("check: the target nodes differ from expected.csv")
        return False
    p = p.loc[en.index]
    stack_col, kmh_col = expected_columns(ens)
    print(f"\ncheck against {SAMPLE.name}/expected.csv ({len(en)} target nodes, "
          f"{len(es)} served segments):")
    for m in ens.members:
        a, b = p[f"pred_{m}"].to_numpy(), en[f"pred_{m}"].to_numpy()
        if m.startswith("gbm"):
            good = bool(np.array_equal(a, b))
            print(f"  {m:24s} max|diff| {np.abs(a - b).max():.3g}  {'EXACT' if good else 'FAIL'}")
        else:
            r = rel(a, b)
            good = r <= TOL
            print(f"  {m:24s} max rel diff {r:.2g}  {'OK' if good else 'FAIL'}")
        ok &= good
    for lab, a, b in (("stack (log v85)", p["pred_stack"], en[stack_col]),
                      ("node v85 km/h", p["v85_pred_kmh"], en[kmh_col]),
                      ("served v85 km/h", served.set_index("old_id").loc[es.index, "v85_pred_kmh"],
                       es[kmh_col])):
        r = rel(a, b)
        ok &= r <= TOL
        print(f"  {lab:24s} max rel diff {r:.2g}  {'OK' if r <= TOL else 'FAIL'}")
    return bool(ok)


def vision_from_images(nodes: pd.DataFrame, edges: dict, ens, exp: pd.DataFrame) -> int:
    """Recompute the Thai targets' v_* features (and, when the stack has the DINOv3 member,
    the classifier scores a_*) with safespeed.vision from the shipped images' cached
    embeddings, detections and DINOv3 features; they must equal the node features."""
    vcols = [c for c in ens.feature_columns() if c.startswith(("v_", "a_"))]
    if not vcols:
        print(f"\n--vision-from-images: NOT VERIFIED - the '{ens.stack_name}' stack uses no image "
              "features, so there is nothing to chain.")
        return 2
    n_a = sum(c.startswith("a_") for c in vcols)      # DINOv3 classifier scores: with_dinov3 only
    try:
        from safespeed import vision
        need = ("predict_segments", "read_images_csv", "read_segments_csv", "load_embeddings") \
            + (("load_dinov3_features",) if n_a else ())
        if not all(hasattr(vision, f) for f in need):
            raise ImportError("safespeed.vision lacks " + ", ".join(f for f in need
                                                                 if not hasattr(vision, f)))
        files = [VSAMPLE / f for f in ("images.csv", "segments.csv", "clip_embeddings.npz",
                                       "detections.jsonl")
                 + (("dinov3_features.npz",) if n_a else ())]
        absent = [f.name for f in files if not f.exists()]
        if absent:
            raise FileNotFoundError(", ".join(absent))
    except Exception as e:  # noqa: BLE001
        print(f"\n--vision-from-images: NOT VERIFIED - the vision part is not available ({e}).")
        return 2
    try:
        vis = vision.predict_segments(vision.read_images_csv(files[0]),
                                      vision.read_segments_csv(files[1]),
                                      vision.load_embeddings(files[2]), detections=str(files[3]),
                                      dinov3=vision.load_dinov3_features(files[4]) if n_a else None)
    except (FileNotFoundError, vision.VisionInputError) as e:
        print(f"\n--vision-from-images: NOT VERIFIED - the vision part could not run ({e})")
        return 2
    vis = vis.set_index("id")
    missing = [c for c in vcols if c not in vis.columns]
    if missing:
        print(f"\n--vision-from-images: FAILED - the vision part does not output "
              f"{', '.join(missing[:5])}")
        return 1
    tgt = nodes[(nodes["is_target"] == 1) & nodes["old_id"].isin(vis.index)]
    print(f"\n--vision-from-images: {len(vis)} segments from {VSAMPLE.name}/ -> "
          f"{len(tgt)} target nodes of {tgt['old_id'].nunique()} served ids; "
          f"{len(vcols) - n_a} v_* features + has_vision"
          + (f" + {n_a} classifier scores a_*" if n_a else ""))
    if tgt.empty:
        print("  no target node belongs to a vision-sample segment: nothing to chain")
        return 1
    new = nodes.copy()
    worst_emb, worst_a, bad = 0.0, 0.0, []
    for i, r in tgt.iterrows():
        v = vis.loc[r["old_id"]]
        got = pd.to_numeric(v[vcols], errors="coerce").to_numpy(np.float64)
        want = r[vcols].to_numpy(np.float64)
        for c, a, b in zip(vcols, got, want):
            if c.startswith("v_emb_"):
                d = abs(a - b)
                worst_emb = max(worst_emb, d)
                if not d <= EMB_TOL:
                    bad.append((r["old_id"], c, a, b))
            elif c.startswith("a_"):
                d = abs(a - b)
                worst_a = max(worst_a, d)
                if not d <= A_TOL:
                    bad.append((r["old_id"], c, a, b))
            elif not ((np.isnan(a) and np.isnan(b)) or a == b):
                bad.append((r["old_id"], c, a, b))
        hv = float(int(v["v_n_forward"]) >= 1)
        if hv != float(r["has_vision"]):
            bad.append((r["old_id"], "has_vision", hv, r["has_vision"]))
        new.loc[i, vcols] = got
        new.loc[i, "has_vision"] = hv
    n_emb = sum(c.startswith("v_emb_") for c in vcols)
    exact = not [b for b in bad if not b[1].startswith(("v_emb_", "a_"))]
    print(f"  features: {len(vcols) - n_emb - n_a} attribute columns + has_vision "
          f"{'EXACT' if exact else 'DIFFER'}; {n_emb} embedding PCs max|diff| {worst_emb:.1e} "
          f"(tolerance {EMB_TOL:g})"
          + (f"; {n_a} classifier scores max|diff| {worst_a:.1e} (tolerance {A_TOL:g})"
             if n_a else ""))
    for b in bad[:10]:
        print(f"    {b[0]} {b[1]}: image route {b[2]!r} vs node {b[3]!r}")
    pred = ens.predict(new, edges)
    en = exp[exp["level"] == "node"].set_index("tt_id")
    p = pred[pred["is_target"] == 1].set_index("tt_id").loc[en.index]
    worst = max(rel(p[f"pred_{m}"], en[f"pred_{m}"]) for m in ens.members)
    ws = rel(p["v85_pred_kmh"], en[expected_columns(ens)[1]])
    ok = not bad and worst <= TOL and ws <= TOL
    print(f"  predictions from the image-route features: members max rel diff {worst:.2g}, "
          f"v85 km/h {ws:.2g}  ->  {'CHAINED: PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the SafeSpeed research ensemble.")
    ap.add_argument("--graph", default=str(SAMPLE),
                    help="folder with nodes.csv, edges_topological.csv, edges_knn.csv "
                         "(default: data/graph_sample/)")
    ap.add_argument("--stack", default=ensemble.DEFAULT_STACK, choices=ensemble.STACKS,
                    help="'default' (7 members, no DINOv3 classifier member), 'with_dinov3' "
                         "(8: adds the DINOv3 classifier member, whose small gain is matched "
                         "by a placebo) or 'without_images' (4: no image features)")
    ap.add_argument("--output", default=str(HERE / "outputs" / "ensemble_predictions.csv"),
                    help="where to write the results (default: outputs/ensemble_predictions.csv)")
    ap.add_argument("--check", action="store_true", help="compare with the graph's expected.csv")
    ap.add_argument("--vision-from-images", action="store_true",
                    help="recompute the Thai target pieces' image features from "
                         "data/vision_sample/ and check that they match the node features")
    args = ap.parse_args()

    try:
        ens = ensemble.load(stack=args.stack)
    except FileNotFoundError as e:
        print(f"error: file not found: {e.filename or e}", file=sys.stderr)
        return 2
    try:
        nodes, edges = ensemble.read_graph(args.graph)
        pred = ens.predict(nodes, edges)
    except ensemble.GraphInputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"error: file not found: {e.filename} (--graph needs nodes.csv, "
              "edges_topological.csv and edges_knn.csv)", file=sys.stderr)
        return 2
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"error: a file in {args.graph} is not a readable UTF-8 CSV: {e}", file=sys.stderr)
        return 2
    exp_path = Path(args.graph) / "expected.csv"
    exp = (pd.read_csv(exp_path, float_precision="round_trip",
                       dtype={"tt_id": str, "old_id": str}) if exp_path.exists() else None)
    if exp is not None and "v85_measured" not in nodes.columns:
        meas = exp[exp["level"] == "node"].set_index("tt_id")["v85_measured"]
        nodes["v85_measured"] = nodes["tt_id"].map(meas)
    served = ens.served(nodes, pred)
    tg = pred[pred["is_target"] == 1] if "is_target" in pred.columns else pred
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(tg, out)                  # float32 columns rounded (safespeed.CSV_DECIMALS)
    try:
        shown = os.path.relpath(out)
    except ValueError:
        shown = str(out)
    print(f"ensemble: {len(nodes)} nodes ({len(tg)} targets), edges "
          + ", ".join(f"{g} {len(e[0])}" for g, e in edges.items()) + f" -> {shown}")
    show(served, ens)

    rc = 0
    if args.check:
        if exp is None:
            print("check: no expected.csv next to the graph")
            rc = 1
        else:
            good = check(pred, served, exp, ens)
            print(f"\ncheck: {'PASSED' if good else 'FAILED'} - trees exact, GraphSAGE / stack / "
                  f"km/h within {TOL:g} relative of {exp['provenance'].iloc[0].split(':')[0]} "
                  "torch/scikit-learn results.")
            rc = 0 if good else 1
    if args.vision_from_images:
        if exp is None:
            print("--vision-from-images needs the sample's expected.csv")
            rc = 1
        else:
            rc = max(rc, vision_from_images(nodes, edges, ens, exp))
    return rc


if __name__ == "__main__":
    sys.exit(main())
