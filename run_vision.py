#!/usr/bin/env python3
"""Run the SafeSpeed vision model: road attributes, invited speed and harm from street images.

    python3 run_vision.py                 # the sample (3 Thai segments, 7 images), cached features
    python3 run_vision.py --check         # ... and compare with data/vision_sample/expected_vision.csv
    python3 run_vision.py --with-dinov3   # also run the DINOv3 ThaiRAP classifier (a_* columns)
    python3 run_vision.py --images my_images.csv --segments my_segments.csv \\
        --embeddings my_emb.npz --detections my_detections.jsonl --output my_vision.csv
    python3 run_vision.py --images my_images.csv --segments my_segments.csv \\
        --image-dir my_jpgs/ --embed-to my_emb.npz ...   # new images: embed them first

Writes outputs/vision_predictions.csv (one row per segment) and
outputs/vision_images.csv (one row per segment point and image: probe codes,
P(divided), expected lanes, detections; with --with-dinov3 also the classifier's
probabilities a_*). Bad input rows are refused with a list of every problem (exit
code 2).

Needs numpy and pandas (Pillow, torch and transformers only to embed images; timm
too for DINOv3). Everything in weights/vision_thairap/ is trained on ThaiRAP labels
and is non-commercial (CC BY-NC 4.0): see weights/vision_thairap/LICENSE.txt.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True          # keep the repository free of __pycache__

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "data" / "vision_sample"
sys.path.insert(0, str(HERE))
from safespeed import vision, write_csv  # noqa: E402

#: v_emb_* are float32 PCA scores. A float32 matrix product is rounded
#: differently for different matrix sizes (BLAS blocking), so the sample's 7
#: images reproduce the values of SafeSpeed's full build (computed on 45,294
#: images at once) to about 1e-7, not bit for bit. They are compared with this absolute tolerance;
#: every other column, and every prediction made from them, must match exactly.
VEMB_TOL = 1e-6
#: a_* are float32 means of float32 probabilities. The package repeats the
#: arithmetic of SafeSpeed's full build and reproduces them bit for bit here;
#: the tolerance only allows for a last-digit difference in another numpy/BLAS
#: build.
A_TOL = 1e-6


def compare(out: pd.DataFrame, exp: pd.DataFrame) -> tuple[dict, list, dict]:
    a = out.drop_duplicates("id").set_index("id")
    b = exp.drop_duplicates("id").set_index("id")
    common = [i for i in a.index if i in b.index]
    bad, emb_d = {}, {"v_emb_": 0.0, "a_": 0.0}
    cols = [c for c in a.columns if c in b.columns]
    for c in cols:
        x, y = a.loc[common, c], b.loc[common, c]
        num = (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)
               and not pd.api.types.is_bool_dtype(x) and not pd.api.types.is_bool_dtype(y))
        if num:
            xf, yf = x.astype(float).to_numpy(), y.astype(float).to_numpy()
            if c.startswith(("v_emb_", "a_")):
                k = "v_emb_" if c.startswith("v_emb_") else "a_"
                diff = ~np.isclose(xf, yf, rtol=0, atol=VEMB_TOL if k == "v_emb_" else A_TOL,
                                   equal_nan=True)
                d = np.abs(xf - yf)
                emb_d[k] = max(emb_d[k], float(np.nanmax(d)) if np.isfinite(d).any() else 0.0)
            else:
                diff = ~np.isclose(xf, yf, rtol=1e-9, atol=1e-9, equal_nan=True)
        else:
            diff = (x.astype(object).where(x.notna(), "").astype(str).to_numpy()
                    != y.astype(object).where(y.notna(), "").astype(str).to_numpy())
        if diff.any():
            bad[c] = int(diff.sum())
    missing = [c for c in a.columns if c not in b.columns]
    return bad, missing + [i for i in a.index if i not in b.index], emb_d


def summary(pred: pd.DataFrame, segs: pd.DataFrame | None) -> None:
    meas = None
    if segs is not None and "operating_speed" in segs.columns:
        meas = pd.to_numeric(segs.drop_duplicates("id").set_index("id")["operating_speed"],
                             errors="coerce")
    print(f"\n{'segment':<10} {'imgs':>4} {'rung':>5}  {'who is at risk':<34} "
          f"{'v50':>5} {'v85':>5} {'meas.':>5} {'harm':>6}  top factor")
    for r in pred.itertuples():
        m = meas.get(r.id, np.nan) if meas is not None else np.nan
        rung = "-" if pd.isna(r.vis_safe_speed) else f"{r.vis_safe_speed:.0f}"

        def f(x, fmt):
            return "-" if pd.isna(x) else format(x, fmt)
        print(f"{r.id:<10} {r.v_n_forward:>4} {rung:>5}  {str(r.vis_safe_speed_reason or '-'):<34} "
              f"{f(r.vis_pred_v50, '.1f'):>5} {f(r.vis_pred_v85, '.1f'):>5} "
              f"{f(m, '.1f'):>5} {f(r.vis_harm, '.3f'):>6}  "
              f"{r.vis_speed_top_factor if isinstance(r.vis_speed_top_factor, str) else '-'}")
    print("\nrung = survivable speed the pictured road allows (km/h); v50/v85 = speed the road "
          "in the photos invites (km/h);\nmeas. = measured TomTom v85; harm = P(killed or "
          "seriously injured) for the rung's road user at those speeds.")
    acols = [c for c in pred.columns if c.startswith("a_")]
    if acols:
        n_attr = len({c[2:].rsplit("__", 1)[0] for c in acols})
        filled = int(pred[acols].notna().all(axis=1).sum())
        print(f"DINOv3 ThaiRAP classifier: {len(acols)} a_* probabilities ({n_attr} design "
              f"attributes) for {filled} of {len(pred)} segments; per image in the images CSV.")
    else:
        print("DINOv3 ThaiRAP classifier: not run (--with-dinov3 adds its a_* columns).")


def clip_embedder():
    """The CLIP embedder, loaded before any image is read. Without the packages in
    requirements-vision.txt, or when the backbone cannot be downloaded, stop with a plain
    message (exit code 1) instead of a traceback."""
    from safespeed.vision import embed
    try:
        return embed.Embedder()
    except (ImportError, RuntimeError) as e:
        sys.exit(f"--embed-to: {e}")


def dinov3_path(dest: Path) -> Path:
    """my_emb.npz -> my_emb.dinov3.npz"""
    return dest.with_name(dest.stem + ".dinov3.npz")


def embed_to(images: pd.DataFrame, image_dir: Path, dest: Path, dinov3: bool):
    """Embed the images the model will use and save them in the package's npz formats:
    CLIP embeddings to `dest`; with `dinov3`, DINOv3 features to dest with .dinov3.npz."""
    from safespeed.vision import embed
    kept = vision.filter_images(images)
    dino = None
    print(f"--embed-to: embedding {kept.image_id.nunique()} images from {image_dir} "
          f"with CLIP ViT-L/14 ({embed.MODEL_ID} @ {embed.REVISION[:12]}) ...")
    clip = clip_embedder()                  # fail early, before any image is read
    if dinov3:                              # ... and before the long CLIP pass
        try:
            dino_model = embed.DinoV3Embedder()
        except (ImportError, RuntimeError) as e:
            sys.exit(f"--embed-to: {e}\nInstall requirements-vision.txt (timm is needed for "
                     "DINOv3), or leave out --with-dinov3 to embed with CLIP only.")
    ids, E = embed.embed_images(kept, image_dir, clip)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest, ids=np.asarray(ids, dtype=str), emb=E.astype(np.float16))  # float16, as SafeSpeed's full build caches them; no pickle
    print(f"  saved {len(ids)} embeddings -> {dest}")
    if dinov3:
        print(f"  DINOv3 ViT-L/16 features ({embed.DINOV3_HUB_ID} @ {embed.DINOV3_REVISION[:12]}) ...")
        dids, F = embed.dinov3_features(kept, image_dir, dino_model)
        np.savez(dinov3_path(dest), ids=np.asarray(dids, dtype=str), X=F.astype(np.float16),
                 model=np.array(embed.DINOV3_HUB_ID), revision=np.array(embed.DINOV3_REVISION))
        print(f"  saved {len(dids)} DINOv3 features -> {dinov3_path(dest)}")
        dino = vision.load_dinov3_features(dinov3_path(dest))
    return vision.load_embeddings(dest), dino


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the SafeSpeed vision model (Thailand).")
    ap.add_argument("--images", default=str(SAMPLE / "images.csv"),
                    help="CSV with one row per (segment point, image); columns in COLUMNS.md "
                         "(default: the sample)")
    ap.add_argument("--segments", default=str(SAMPLE / "segments.csv"),
                    help="CSV with one row per segment: id, country (TH), v_n_points "
                         "(default: the sample)")
    ap.add_argument("--embeddings", default=str(SAMPLE / "clip_embeddings.npz"),
                    help="the images' CLIP embeddings (.npz with ids and emb), made by "
                         "--embed-to (default: the sample's)")
    ap.add_argument("--detections", default=str(SAMPLE / "detections.jsonl"),
                    help="Mapillary detections .jsonl ('' for none)")
    ap.add_argument("--image-dir", default=str(SAMPLE / "images"),
                    help="folder with <image_id>.jpg, for --embed-to")
    ap.add_argument("--output", default=str(HERE / "outputs" / "vision_predictions.csv"),
                    help="per-segment results; the per-image table is written next to it "
                         "(default: outputs/vision_predictions.csv)")
    ap.add_argument("--check", action="store_true",
                    help="compare with data/vision_sample/expected_vision.csv")
    ap.add_argument("--embed-to", metavar="NEW.npz",
                    help="embed the (filtered) images in --image-dir with CLIP (-> NEW.npz; "
                         "with --with-dinov3 also DINOv3 -> NEW.dinov3.npz) and use them "
                         "(needs requirements-vision.txt; downloads CLIP, about "
                         "1.7 GB, once, and with --with-dinov3 DINOv3, about 1.2 GB)")
    ap.add_argument("--with-dinov3", action="store_true",
                    help="also run the DINOv3 ThaiRAP classifier and add its a_* columns (only "
                         "the ensemble's with_dinov3 stack uses them)")
    ap.add_argument("--dinov3-features", default=str(SAMPLE / "dinov3_features.npz"),
                    help="with --with-dinov3: cached DINOv3 features .npz (default: the "
                         "sample's)")
    args = ap.parse_args()
    use_dino = args.with_dinov3

    t0 = time.time()
    try:
        images = vision.read_images_csv(args.images)
        segs = vision.read_segments_csv(args.segments) if args.segments else None
        dino = None
        if args.embed_to:
            emb, dino = embed_to(images, Path(args.image_dir), Path(args.embed_to), use_dino)
        else:
            emb = vision.load_embeddings(args.embeddings)
            if use_dino:
                dino = vision.load_dinov3_features(args.dinov3_features)
        det = args.detections or None
        pred, per_img = vision.predict_segments(images, segs, emb, detections=det,
                                                return_images=True, dinov3=dino)
    except vision.VisionInputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"error: file not found: {e.filename or e}", file=sys.stderr)
        return 2
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"error: an input is not a readable UTF-8 CSV: {e}", file=sys.stderr)
        return 2
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(pred, out)                    # float32 columns rounded (safespeed.CSV_DECIMALS)
    write_csv(per_img, out.with_name(out.stem.replace("predictions", "images") + out.suffix)
              if "predictions" in out.stem else out.with_name(out.stem + "_images.csv"))
    try:
        shown = os.path.relpath(out)
    except ValueError:                      # another drive on Windows
        shown = str(out)
    print(f"vision: {len(pred)} segments, {per_img.image_id.nunique()} images -> {shown} "
          f"({time.time() - t0:.1f} s)")
    summary(pred, segs)

    rc = 0
    if args.check:
        exp = pd.read_csv(SAMPLE / "expected_vision.csv", float_precision="round_trip",
                          dtype={"id": str})
        bad, missing, emb_d = compare(pred, exp)
        is_sample = Path(args.images).resolve() == (SAMPLE / "images.csv").resolve()
        if missing:
            print(f"\ncheck: not in the expected results: {', '.join(map(str, missing[:8]))}")
        if bad or (missing and is_sample):
            shown = dict(list(bad.items())[:6])
            more = f" and {len(bad) - len(shown)} more" if len(bad) > len(shown) else ""
            print(f"\ncheck: FAILED - columns that differ (rows): {shown or '-'}{more}")
            rc = 1
        else:
            n_emb = sum(c.startswith("v_emb_") for c in pred.columns)
            n_a = sum(c.startswith("a_") for c in pred.columns)
            print(f"\ncheck: PASSED - {pred.shape[1] - 1 - n_emb - n_a} result columns match the "
                  f"expected results exactly (rtol 1e-9) for {len(pred)} segments; the "
                  f"{n_emb} embedding components v_emb_* agree to {emb_d['v_emb_']:.1e} "
                  f"(float32, tolerance {VEMB_TOL:g})"
                  + (f" and the {n_a} classifier scores a_* to {emb_d['a_']:.1e} (float32, "
                     f"tolerance {A_TOL:g})." if n_a else
                     ". The DINOv3 classifier was not run; --with-dinov3 adds and checks its "
                     "a_* columns."))
    return rc


if __name__ == "__main__":
    sys.exit(main())
