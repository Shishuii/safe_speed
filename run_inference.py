#!/usr/bin/env python3
"""Predict design speeds for road segments with the SafeSpeed model.

    python3 run_inference.py                    # predicts data/sample_segments.csv
    python3 run_inference.py --check            # ... and compares with the expected results
    python3 run_inference.py --input my_roads.csv --output my_predictions.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True          # keep the shared folder free of __pycache__

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "data" / "sample_segments.csv"
EXPECTED = HERE / "data" / "expected_predictions.csv"
sys.path.insert(0, str(HERE))
from safespeed._assets import require_assets  # noqa: E402
require_assets(HERE)            # data/ and weights/ come from the Google Drive download
import safespeed  # noqa: E402


def check(out: pd.DataFrame, is_sample: bool) -> bool:
    """Compare every output column with the expected results, row by row (by id)."""
    exp = safespeed.read_csv(EXPECTED)
    ok = True
    dup = sorted(set(out["id"][out["id"].duplicated()]))
    if dup:
        print(f"check: ids repeated in the input, first occurrence compared: {', '.join(dup[:5])}")
    a = out.drop_duplicates("id").set_index("id")
    b = exp.drop_duplicates("id").set_index("id")
    common = [i for i in a.index if i in b.index]
    unchecked = len(a) - len(common)
    if not common:
        print("check: none of the input ids are in the expected results; nothing to compare.")
        return False
    if unchecked:
        print(f"check: {unchecked} input rows have no expected result and were not checked.")
        ok = not is_sample                  # the shipped sample must be checked in full
    absent = [c for c in a.columns if c not in b.columns]
    if absent:
        print(f"check: the expected file lacks columns: {', '.join(absent)}")
        ok = False
    bad = {}
    for c in [c for c in a.columns if c in b.columns]:
        x, y = a.loc[common, c], b.loc[common, c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) \
                and not pd.api.types.is_bool_dtype(x):
            diff = ~np.isclose(x.astype(float), y.astype(float), rtol=1e-9, atol=1e-9,
                               equal_nan=True)
        else:
            diff = x.astype(object).where(x.notna(), "").astype(str) != \
                y.astype(object).where(y.notna(), "").astype(str)
        if diff.any():
            bad[c] = int(diff.sum())
    if bad:
        print(f"check: FAILED - columns that differ (rows): {bad}")
        return False
    if not ok:
        print("check: FAILED")
        return False
    print(f"check: PASSED - all {a.shape[1]} result columns (plus id) match the expected "
          f"results for {len(common)} segments.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Predict design speeds with the SafeSpeed model.")
    ap.add_argument("--input", default=str(SAMPLE),
                    help="CSV with one row per road segment (default: the sample)")
    ap.add_argument("--output", default=str(HERE / "outputs" / "predictions.csv"),
                    help="where to write the results (default: outputs/predictions.csv)")
    ap.add_argument("--check", action="store_true",
                    help="compare the results with data/expected_predictions.csv")
    args = ap.parse_args()

    try:
        df = safespeed.read_csv(args.input)
        t0 = time.time()
        out = safespeed.predict(df)
        secs = time.time() - t0
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
    except safespeed.InputError as e:
        msg = str(e)
        if "missing required columns" in msg and len(df.columns) == 1 and ";" in df.columns[0]:
            msg += "\n  The file looks semicolon-separated; save it as a comma-separated CSV."
        print(f"error: {msg}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"error: file not found: {e.filename}", file=sys.stderr)
        return 2
    except IsADirectoryError as e:
        print(f"error: {e.filename} is a folder; give a file name", file=sys.stderr)
        return 2
    except UnicodeDecodeError:
        print(f"error: {args.input} is not UTF-8 text; save it as a UTF-8 CSV "
              "(an Excel file must be exported to CSV first)", file=sys.stderr)
        return 2
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"error: {args.input} is not a readable CSV: {e}", file=sys.stderr)
        return 2

    shown = Path(args.output).resolve()
    try:
        shown = shown.relative_to(Path.cwd())
    except ValueError:
        pass
    print(f"Predicted {len(out)} road segments in {secs:.1f} s -> {shown}\n")
    names = {"MH": "Maharashtra", "TH": "Thailand"}
    def kmh(x, w):
        return f"{'-':>{w}}     " if pd.isna(x) else f"{x:>{w}.0f} km/h"

    print("Medians       segments   posted limit   design v85   measured v85")
    for c, g in out.groupby("country"):
        posted = pd.to_numeric(df.loc[g.index, "posted_limit"], errors="coerce").median()
        meas = pd.to_numeric(df.loc[g.index, "operating_speed"], errors="coerce").median() \
            if "operating_speed" in df.columns else float("nan")
        print(f"  {names.get(c, c):<12} {len(g):>6}   {kmh(posted, 7)}   "
              f"{kmh(g['design_speed_v85'].median(), 6)}   {kmh(meas, 7)}")
    print("\nFactor that moves the design speed most (speed_top_factor):")
    for f, n in out["speed_top_factor"].value_counts().items():
        print(f"  {n:>5}  {f}")
    print("\nExamples:")
    for _, r in out.head(3).iterrows():
        print(f"  {r['id']:<9} v50 {r['design_speed_v50']:.0f} / v85 {r['design_speed_v85']:.0f} km/h "
              f"(reference road: {r['speed_attr_base_kmh']:.0f}); largest effect: "
              f"{r['speed_top_factor'].lower()}")

    if args.check:
        print()
        is_sample = Path(args.input).resolve() == SAMPLE.resolve()
        return 0 if check(out, is_sample) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
