"""OPTIONAL: image embeddings from pixels (CLIP ViT-L/14 and DINOv3 ViT-L/16).

Needs torch and transformers (see requirements-vision.txt) and downloads the
CLIP weights from Hugging Face on first use (openai/clip-vit-large-patch14 at a
pinned revision, about 1.7 GB; the weights are not redistributed with this
package). Not needed for the sample or for `run_vision.py --check`, which use
the cached embeddings.

One L2-normalised 768-d vector per image: CLIP's own preprocessing (shortest
side to 224 px, bicubic; centre crop 224 x 224; CLIP mean/std), the image
projection (`get_image_features`), float16 on a GPU, float32 on a CPU, batches
of 64. A panorama is first cropped to a 1024 x 768 perspective view (100 degree
field of view, level horizon) facing along the road.

Recomputed embeddings are NOT bit-identical to the cached ones in general. The
cache was made with transformers 4.57 on a CUDA GPU in fp16, batch 64; other
library versions, a CPU, or another batch size change the vectors slightly
(measured: cosine >= 0.9997, and about 0.5 % of ThaiRAP probe codes flip on
random images). Treat a recomputation as a tolerance check, not an exact one.

DINOv3 features (for the ThaiRAP classifier, safespeed/vision/classifier.py)
need torch and timm >= 1.0.20 and download the backbone from Hugging Face
(timm/vit_large_patch16_dinov3.lvd1689m at a pinned revision, about 1.2 GB,
DINOv3 License; the weights are not redistributed with this package). The image
(a panorama first cropped along the road, as for CLIP) is centre-cropped to
16:9, resized to 512 x 288 (bicubic) and normalised with the ImageNet mean/std;
the backbone runs under fp16 autocast on a GPU (fp32 on a CPU), batches of 32.
Five views of its output tokens are pooled and concatenated (5 x 1,024): the
first token, the mean of all patch tokens, and the mean of the left, centre and
right thirds of the patch grid. The features are stored as float16, as the
pipeline cached them. The same caveat applies: a recomputation is a tolerance
check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_ID = "openai/clip-vit-large-patch14"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"

DINOV3_MODEL = "vit_large_patch16_dinov3.lvd1689m"            # timm model name
DINOV3_HUB_ID = "timm/vit_large_patch16_dinov3.lvd1689m"      # Hugging Face repo
DINOV3_REVISION = "30c1109559f65dea34316b0d4842d35c5771fe11"
DINOV3_INPUT_WH = (512, 288)                                  # 32 x 18 patches of 16 px
DINOV3_BATCH = 32


# ------------------------------------------------------------ panoramas
def perspective(eq, yaw_deg: float, fov_deg: float = 100.0, pitch_deg: float = 0.0,
                out_w: int = 1024, out_h: int = 768):
    """Perspective crop of an equirectangular PIL image. yaw 0 = image centre."""
    from PIL import Image
    src = np.asarray(eq.convert("RGB"))
    H, W = src.shape[:2]
    f = 0.5 * out_w / np.tan(np.radians(fov_deg) / 2)
    x = np.arange(out_w) - (out_w - 1) / 2
    y = np.arange(out_h) - (out_h - 1) / 2
    xx, yy = np.meshgrid(x, y)
    v = np.stack([xx, yy, np.full_like(xx, f)], axis=-1)       # x right, y down, z forward
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    p, yw = np.radians(pitch_deg), np.radians(yaw_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(yw), 0, np.sin(yw)], [0, 1, 0], [-np.sin(yw), 0, np.cos(yw)]])
    v = v @ (ry @ rx).T
    lon = np.arctan2(v[..., 0], v[..., 2])
    lat = np.arcsin(np.clip(v[..., 1], -1, 1))
    u = ((lon / (2 * np.pi) + 0.5) * W).astype(np.int64) % W
    w = ((lat / np.pi + 0.5) * H).clip(0, H - 1).astype(np.int64)
    return Image.fromarray(src[w, u])


def road_yaw(compass_angle: float, road_bearing: float) -> float:
    """Yaw offset (deg) from the panorama centre to the nearer road direction."""
    if not np.isfinite(road_bearing):
        return 0.0
    best = None
    for b in (road_bearing, (road_bearing + 180.0) % 360.0):
        d = (b - compass_angle + 180.0) % 360.0 - 180.0
        if best is None or abs(d) < abs(best):
            best = d
    return float(best)


def load_image(path, is_pano: bool = False, compass_angle: float = np.nan,
               road_bearing: float = np.nan):
    """An image as CLIP sees it: RGB; a panorama cropped to face along the road."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if is_pano:
        im = perspective(im, road_yaw(float(compass_angle), float(road_bearing)))
    return im


# ------------------------------------------------------------ CLIP
class Embedder:
    def __init__(self, device: str | None = None, revision: str = REVISION):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:            # pragma: no cover
            raise ImportError("recomputing embeddings needs torch and transformers: "
                              "pip install torch transformers") from e
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.dev.startswith("cuda") else torch.float32
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"   # renamed in 4.56
        model = CLIPModel.from_pretrained(MODEL_ID, revision=revision, **{kw: self.dtype})
        self.model = model.to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained(MODEL_ID, revision=revision)

    def __call__(self, images, batch: int = 64) -> np.ndarray:
        """`images`: a list of PIL images. Returns float16 (n x 768), L2-normalised."""
        torch = self.torch
        out = []
        for i in range(0, len(images), batch):
            ims = images[i:i + batch]
            with torch.no_grad():
                x = self.proc(images=ims, return_tensors="pt")["pixel_values"]
                e = self.model.get_image_features(pixel_values=x.to(self.dev, self.dtype))
                e = e if torch.is_tensor(e) else e.pooler_output      # transformers 4 vs 5
                e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.float().cpu().numpy().astype(np.float16))
        return np.concatenate(out) if out else np.zeros((0, 768), np.float16)


def _load_first_images(images_table, image_dir: Path):
    """(ids, PIL images) for every unique image, a panorama cropped along the road
    bearing of its first row."""
    from .aggregate import parse_bool
    first = images_table.drop_duplicates("image_id")
    first = first.assign(_pano=parse_bool(first["is_pano"], "is_pano")
                         if "is_pano" in first.columns else False)
    ims = [load_image(Path(image_dir) / f"{r['image_id']}.jpg",
                      bool(r["_pano"]),
                      float(r.get("compass_angle", np.nan)),
                      float(r.get("road_bearing", np.nan)))
           for r in first.to_dict("records")]
    return first["image_id"].astype(str).to_numpy(), ims


def embed_images(images_table, image_dir: Path, embedder: Embedder | None = None,
                 batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Embed every unique image of an images table (files <image_dir>/<image_id>.jpg).
    Pass the table AFTER the image filter (safespeed.vision.filter_images), as the
    pipeline did: a panorama is cropped along the road bearing of its first kept row.
    Returns (ids, float32 embeddings)."""
    ids, ims = _load_first_images(images_table, image_dir)
    emb = (embedder or Embedder())(ims, batch=batch)
    return ids, emb.astype(np.float32)


# ------------------------------------------------------------ DINOv3
def to_band(img):
    """Centre crop of a PIL image to 16:9 (the classifier's training shape)."""
    w, h = img.size
    th = int(round(w * 9 / 16))
    if th >= h:
        tw = int(round(h * 16 / 9))
        x0 = (w - tw) // 2
        return img.crop((x0, 0, x0 + tw, h))
    y0 = (h - th) // 2
    return img.crop((0, y0, w, y0 + th))


class DinoV3Embedder:
    """Frozen DINOv3 ViT-L/16 with the classifier's five pooled views."""

    def __init__(self, device: str | None = None, revision: str = DINOV3_REVISION):
        try:
            import timm
            import torch
        except ImportError as e:
            raise ImportError("DINOv3 features need torch and timm >= 1.0.20: "
                              "pip install -r requirements-vision.txt") from e
        ver = tuple(int(x) for x in (timm.__version__.split(".") + ["0", "0"])[:3]
                    if x.isdigit())
        if ver < (1, 0, 20):
            raise ImportError(f"DINOv3 needs timm >= 1.0.20 (found {timm.__version__})")
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = timm.create_model(DINOV3_MODEL, pretrained=True, num_classes=0,
                                      dynamic_img_size=True,
                                      pretrained_cfg_overlay={"hf_hub_id": f"{DINOV3_HUB_ID}@{revision}"})
        except Exception as e:  # noqa: BLE001 - download, licence or network problems
            raise RuntimeError(f"could not load the DINOv3 backbone {DINOV3_HUB_ID} at revision "
                               f"{revision[:12]} ({type(e).__name__}: {e}). It is downloaded from "
                               "Hugging Face (about 1.2 GB) under the DINOv3 License.") from e
        self.model = model.to(self.dev).eval()
        cfg = timm.data.resolve_model_data_config(self.model)
        self.mean = torch.tensor(cfg["mean"], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=self.dev).view(1, 3, 1, 1)
        self.npre = getattr(self.model, "num_prefix_tokens", 1)
        self.patch = self.model.patch_embed.patch_size[0]
        self.W, self.H = DINOV3_INPUT_WH

    def _tensor(self, ims):
        from PIL import Image
        bicubic = getattr(Image, "Resampling", Image).BICUBIC
        arr = np.stack([np.asarray(to_band(im.convert("RGB")).resize((self.W, self.H), bicubic),
                                   dtype=np.uint8) for im in ims])
        x = self.torch.from_numpy(arr.copy()).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std

    def _pool(self, tokens):
        torch = self.torch
        B, _, D = tokens.shape
        gh, gw = self.H // self.patch, self.W // self.patch
        patches = tokens[:, self.npre:, :]
        grid = patches.reshape(B, gh, gw, D)
        third = gw // 3
        return torch.cat([tokens[:, 0], patches.mean(dim=1),
                          grid[:, :, :third].mean(dim=(1, 2)),
                          grid[:, :, third:gw - third].mean(dim=(1, 2)),
                          grid[:, :, gw - third:].mean(dim=(1, 2))], dim=1)

    def __call__(self, images, batch: int = DINOV3_BATCH) -> np.ndarray:
        """`images`: a list of PIL images. Returns float16 (n x 5,120)."""
        torch = self.torch
        out = []
        for i in range(0, len(images), batch):
            x = self._tensor(images[i:i + batch])
            with torch.no_grad():
                if self.dev.startswith("cuda"):
                    with torch.autocast("cuda", dtype=torch.float16):
                        f = self._pool(self.model.forward_features(x))
                else:
                    f = self._pool(self.model.forward_features(x))
            out.append(f.float().cpu().numpy().astype(np.float16))
        dim = 5 * int(self.model.num_features)
        return np.concatenate(out) if out else np.zeros((0, dim), np.float16)


def dinov3_features(images_table, image_dir: Path, embedder: DinoV3Embedder | None = None,
                    batch: int = DINOV3_BATCH) -> tuple[np.ndarray, np.ndarray]:
    """DINOv3 features of every unique image of an (already filtered) images table.
    Returns (ids, float16 features), the format of dinov3_features.npz."""
    ids, ims = _load_first_images(images_table, image_dir)
    return ids, (embedder or DinoV3Embedder())(ims, batch=batch)
