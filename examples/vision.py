"""Example 2: the vision model on the sample street images (Thailand).

Run from the top folder:   python examples/vision.py
Writes outputs/examples/vision.csv (one row per road segment).
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                  # keep the folder free of __pycache__
HERE = Path(__file__).resolve().parent.parent   # the top folder, with run_inference.py
sys.path.insert(0, str(HERE))
from safespeed import vision, write_csv  # noqa: E402

S = HERE / "data" / "vision_sample"
images = vision.read_images_csv(S / "images.csv")         # one row per image: where, when, which way
segments = vision.read_segments_csv(S / "segments.csv")   # one row per road segment
clip = vision.load_embeddings(S / "clip_embeddings.npz")  # the images' CLIP vectors, computed in advance

# Mapillary's detections (people, riders, crossings ...) show which road users the road exposes.
# To add the ThaiRAP classifier's a_* columns, also pass
# dinov3=vision.load_dinov3_features(S / "dinov3_features.npz").
pred = vision.predict_segments(images, segments, clip, detections=str(S / "detections.jsonl"))

cols = ["id", "v_n_images", "vis_safe_speed", "vis_pred_v85", "vis_harm"]
print(pred[cols].round(3).to_string(index=False))
for road in pred.itertuples():
    print(f"{road.id}: safe speed {road.vis_safe_speed:.0f} km/h, {road.vis_safe_speed_reason}")

out = HERE / "outputs" / "examples" / "vision.csv"
out.parent.mkdir(parents=True, exist_ok=True)
write_csv(pred, out)    # rounds the float32 columns, so every platform writes the same file
print(f"\nwrote {out.relative_to(HERE)} ({len(pred)} segments)")
