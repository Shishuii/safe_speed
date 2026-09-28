# Attribution: sample street images and detections

## Mapillary images and detections (CC BY-SA 4.0)

The 7 street images in `images/` and the object detections in
`detections.jsonl` come from **Mapillary** (https://www.mapillary.com).

- Images: (c) Mapillary contributors, licensed under **CC BY-SA 4.0**
  (https://creativecommons.org/licenses/by-sa/4.0/). Each image's page, with its
  contributor, is linked below. The files are Mapillary's 1024 px thumbnails,
  unchanged.
- Detections: Mapillary's own hosted detections for these images, retrieved from the
  Mapillary API (`/<image_id>/detections?fields=value`), (c) Mapillary, CC BY-SA 4.0.
- When you show these images or anything derived from them (including detections or
  panorama crops), credit "(c) Mapillary contributors, CC BY-SA 4.0", link to the image
  page, and keep derivatives under CC BY-SA 4.0. Where Mapillary API data is displayed,
  Mapillary's terms also ask for the Mapillary logo and a link to https://www.mapillary.com.

| image_id | segment | captured | Mapillary page |
|---|---|---|---|
| 1282738293748916 | TH-16881 | 2026-01-30 | https://www.mapillary.com/app/?pKey=1282738293748916 |
| 221553129756406 | TH-16881 | 2021-07-12 | https://www.mapillary.com/app/?pKey=221553129756406 |
| 687580706558681 | TH-19351 | 2023-07-20 | https://www.mapillary.com/app/?pKey=687580706558681 |
| 972675657116385 | TH-19351 | 2023-07-20 | https://www.mapillary.com/app/?pKey=972675657116385 |
| 304286381980734 | TH-19351 | 2023-07-20 | https://www.mapillary.com/app/?pKey=304286381980734 |
| 145082377512941 | TH-35263 | 2017-09-29 | https://www.mapillary.com/app/?pKey=145082377512941 |
| 170655508289416 | TH-35263 | 2014-08-24 | https://www.mapillary.com/app/?pKey=170655508289416 |

## OpenStreetMap (ODbL)

The OSM probes (`weights/vision/osm_probes.npz`: divided carriageway, lane count) were
trained with labels derived from OpenStreetMap: (c) OpenStreetMap contributors, available
under the Open Database License (https://www.openstreetmap.org/copyright).

## DINOv3 features

`dinov3_features.npz` holds the cached DINOv3 ViT-L/16 features of the same images
(outputs of Meta's DINOv3 backbone, timm/vit_large_patch16_dinov3.lvd1689m). DINOv3
is used under the DINOv3 License; the full text is in
`weights/vision_thairap/DINOv3_LICENSE.md`. These features are distributed under the
DINOv3 License, whose conditions (summarised in `weights/vision_thairap/LICENSE.txt`)
apply to anyone who passes them on.

## Not included

No ThaiRAP images or ThaiRAP labels are shipped. The probes and the classifier trained
on them are in `weights/vision_thairap/` under their own non-commercial licence (see
LICENSE.txt there). The CLIP and DINOv3 backbone weights are not redistributed;
`safespeed/vision/embed.py` downloads them from Hugging Face (openai/clip-vit-large-patch14,
timm/vit_large_patch16_dinov3.lvd1689m) only if you recompute embeddings.
