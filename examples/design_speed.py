"""Example 1: the design-speed model on the sample roads.

Run from the top folder:   python examples/design_speed.py
Writes outputs/examples/design_speed.csv (one row per road segment).
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                  # keep the folder free of __pycache__
HERE = Path(__file__).resolve().parent.parent   # the top folder, with run_inference.py
sys.path.insert(0, str(HERE))
import safespeed  # noqa: E402

# Read the roads with safespeed.read_csv: it keeps every digit, and the trees need exact inputs.
roads = safespeed.read_csv(HERE / "data" / "sample_segments.csv")

# The speed each road's layout invites (v50, v85), each factor group's share and the expected harm.
pred = safespeed.predict(roads)

# The roads where measured traffic runs furthest above the speed the layout invites.
cols = ["id", "posted_limit", "design_speed_v85", "speed_vs_design_kmh", "speed_top_factor", "harm_model"]
table = pred.merge(roads[["id", "posted_limit"]], on="id")[cols]
print(table.sort_values("speed_vs_design_kmh", ascending=False).head(8).round(3).to_string(index=False))

out = HERE / "outputs" / "examples" / "design_speed.csv"
out.parent.mkdir(parents=True, exist_ok=True)
safespeed.write_csv(pred, out)    # rounds only float32 columns; this model has none
print(f"\nwrote {out.relative_to(HERE)} ({len(pred)} roads)")
