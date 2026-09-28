"""Example 3: the research ensemble on the sample road graph.

Run from the top folder:   python examples/ensemble.py
Writes outputs/examples/ensemble_segments.csv (one row per SafeSpeed segment).
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                  # keep the folder free of __pycache__
HERE = Path(__file__).resolve().parent.parent   # the top folder, with run_inference.py
sys.path.insert(0, str(HERE))
import pandas as pd  # noqa: E402
from safespeed import ensemble, write_csv  # noqa: E402

# Road pieces (nodes) with their features, and the two graphs that join them.
G = HERE / "data" / "graph_sample"
nodes, edges = ensemble.read_graph(G)
# Measured v85 for comparison. The shipped weights were trained on these roads, so this is in-sample.
exp = pd.read_csv(G / "expected.csv", dtype={"tt_id": str}).query("level == 'node'")
nodes["v85_measured"] = nodes["tt_id"].map(exp.set_index("tt_id")["v85_measured"])

ens = ensemble.load()                 # default stack; or stack="with_dinov3" / "without_images"
pieces = ens.predict(nodes, edges)    # one row per piece: each member's log v85 and the stack
segments = ens.served(nodes, pieces)  # one row per SafeSpeed segment, in km/h

print(segments[["old_id", "n_children", "v85_pred_kmh", "v85_measured"]].round(1).to_string(index=False))

out = HERE / "outputs" / "examples" / "ensemble_segments.csv"
out.parent.mkdir(parents=True, exist_ok=True)
write_csv(segments, out)    # rounds the float32 columns, so every platform writes the same file
print(f"\nwrote {out.relative_to(HERE)} ({len(segments)} segments, stack '{ens.stack_name}')")
