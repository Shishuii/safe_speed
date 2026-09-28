# Input and output columns

The sample files in `data/` have every input column and can be copied as templates.

## 1 · Design-speed model

**Input**: one row per road segment (`data/sample_segments.csv`). Required columns
must be present and may be blank only where marked. Optional columns may be left out.

| Column | Required | Meaning | Blank allowed |
|---|---|---|---|
| `id` | yes | segment identifier | no |
| `country` | yes | `MH` (Maharashtra) or `TH` (Thailand) | no |
| `posted_limit` | yes | posted speed limit, km/h; 0 counts as blank | yes |
| `length_km` | yes | segment length, km | no |
| `lat`, `lon` | yes | first point of the segment, WGS-84 | no |
| `poi_vru_raw` | yes | weighted count of places that bring people on foot: 1.0 × schools + 1.0 × public-transport stops + 0.7 × restaurants + 0.5 × hospitals + 0.3 × fuel stations + 0.2 × police and fire stations | no |
| `poi_school` | yes | schools nearby | no |
| `poi_transport` | yes | public-transport stops nearby | no |
| `pop_exposure` | yes | residents within 300 m (WorldPop 2025) | no |
| `pop_density` | yes | mean residents per 100 m cell within 300 m | no |
| `geo_sinuosity` | yes | road length ÷ straight-line length (1 = straight) | no |
| `geo_curvature` | yes | total turning angle per metre, rad/m | no |
| `geo_junction_density` | yes | ends of other network segments within 35 m of either end of this segment, per km (length floored at 50 m) | no |
| `name` | no | road name, passed through | |
| `operating_speed` | no | measured v85, km/h; used only for `speed_vs_design_kmh` | |
| `crash_type` | no | who is most at risk (below); used only for `harm_model`. Blank means `vru_ped` | |

| `crash_type` | Who is at risk | Injury curve | Impact speed |
|---|---|---|---|
| `vru_ped` | person on foot hit by a car | pedestrian | travel speed |
| `vru_ptw` | motorcyclist hit by a car | motorcyclist | travel speed |
| `vru_cycle` | cyclist hit by a car | cyclist | travel speed |
| `head_on` | car occupants in a head-on crash | car occupant | twice the travel speed |
| `side_impact` | car occupant hit in the side at a junction | car occupant | travel speed |
| `motorway` | car occupant on a motorway | car occupant | travel speed |

SafeSpeed sets `crash_type` from its survivable-speed rules: motorways get
`motorway`; other roads go by their GHS-SMOD settlement class (5–6: `vru_ped`; 3–4:
`vru_ptw` with 3 or more lanes, otherwise `vru_ped`; 0–2: `vru_ptw`), and without a
settlement class urban primary and secondary roads get `vru_ped` and the rest
`vru_ptw`.

**Output** (`outputs/predictions.csv`):

| Column | Meaning |
|---|---|
| `id`, `country`, `name` | passed through |
| `design_speed_v50`, `design_speed_v85` | median and 85th-percentile speed the road invites, km/h |
| `speed_vs_design_kmh` | measured v85 minus predicted v85 (blank without `operating_speed`); positive = traffic runs faster than the layout invites |
| `speed_attr_base_kmh` | v85 of a reference road with every input at the country's median |
| `speed_attr_location` … `speed_attr_length` | each factor group's share, in km/h, of the difference from the reference road; the six add up to `design_speed_v85 − speed_attr_base_kmh` |
| `speed_top_factor`, `speed_top_factor_2` | the two factor groups with the largest effect |
| `harm_model` | 0–1: chance that the road user named by `crash_type` is killed or seriously injured in a crash at these speeds |

## 2 · Vision model (Thailand)

**Input** (`data/vision_sample/`):

| File | One row per | Columns |
|---|---|---|
| `images.csv` | image at a segment point | below |
| `segments.csv` | segment | `id`, `country` (`TH`), `v_n_points` (points every 500 m on the segment); optional `operating_speed` (measured v85, shown as `meas.` in the run summary); other columns, such as `road_class`, are ignored |
| `detections.jsonl` | image | Mapillary's detections: `image_id`, `n`, `values` (class names). Without them the people and junction features are blank, and the motorway survivable speed is never chosen |
| `clip_embeddings.npz` | image | `ids`, `emb`: CLIP ViT-L/14 vectors, made by `run_vision.py --embed-to` from `<image_id>.jpg` files |
| `dinov3_features.npz` | image | `ids`, `X`: DINOv3 features, made by `--embed-to --with-dinov3`; only needed for the `a_*` columns |

| `images.csv` column | Meaning |
|---|---|
| `image_id`, `seg_id` | Mapillary image id; the segment the image belongs to (required) |
| `captured_at` or `year` | capture time in ms since 1970, or the capture year (one of them required) |
| `distance_m` | distance from the image to the segment point, m |
| `compass_angle` | the camera's compass heading, degrees |
| `road_bearing` | the road's compass bearing at the point, degrees, measured over 25 m of the road line |
| `heading_diff` | angle between `compass_angle` and `road_bearing`, 0–90°, ignoring travel direction; computed from them when left out |
| `is_pano` | panorama or not: true/false, 1/0 or yes/no |
| `osm_fold` | the image's cross-validation block in the Thai training network; blank elsewhere |
| `key`, `lat`, `lon`, `mapillary_url` | passed through |

A missing optional column skips its filter.

Images are used if captured in 2010 or later, within 40 m of the segment point, and
with the camera within 45° of the road (panoramas are always kept and cropped to
face along it).

**Output** (`outputs/vision_predictions.csv`, one row per segment):

| Column | Meaning |
|---|---|
| `v_*` | image features: shares 0–1, counts, `v_lanes`, and `v_emb_*` (32 principal components of the mean embedding); blank where the images cannot measure them |
| `v_n_images`, `v_n_forward`, `v_n_evidence` | images used, images facing along the road, images with detections |
| `v_n_points`, `v_coverage_frac` | points on the segment; images ÷ points, capped at 1 (roughly the share of the segment the images cover) |
| `v_capture_year`, `v_model`, `v_prompt_hash` | median capture year of the images; the image model; a short hash of the probe settings that made the features |
| `vis_safe_speed`, `vis_crash_type_key`, `vis_safe_speed_reason` | survivable speed (km/h), the road user it protects, and why |
| `vis_motorway_like`, `vis_conflict_head_on`, `vis_conflict_side`, `vis_ped_evidence` | details behind the survivable speed |
| `vis_pred_v50`, `vis_pred_v85` | speed the pictured road invites, km/h |
| `vis_speed_attr_<group>`, `vis_speed_attr_base_kmh`, `vis_speed_top_factor` | each factor group's share of the difference from the reference road, as in model 1 |
| `vis_harm` | 0–1: chance of death or serious injury for that road user at these speeds |
| `vis_status` | `imagery` or `no imagery` |
| `a_<attribute>__<code>` | with `--with-dinov3` only: the DINOv3 ThaiRAP classifier's probability of each iRAP code, averaged over the segment's images |

`outputs/vision_images.csv` has one row per image: the input columns, the image's
features, the ThaiRAP probe codes (`thairap_*`), the divided-road probability
(`p_divided`) and, with `--with-dinov3`, the classifier's `a_*` probabilities.

## 3 · Research ensemble

**Input** (`data/graph_sample/`):

| File | Columns |
|---|---|
| `nodes.csv` | one row per road piece: `idx` (row position), `tt_id`, `old_id` (its SafeSpeed segment; blank for a piece outside the scored segments), `country`, `sample_size` (TomTom probe passes), `is_target` (1 = its whole 2-hop neighbourhood is in the file, so its prediction is valid), then the features listed in `weights/ensemble/features.json` and explained in `weights/ensemble/MODEL_CARD.md`. The columns in `never_blank` of `weights/ensemble/inputs.json` may not be blank |
| `edges_topological.csv`, `edges_knn.csv` | `src`, `dst`: row positions in `nodes.csv` of pieces joined end to end, and of each piece's nearest neighbours |

**Output** (`outputs/ensemble_predictions.csv`, one row per target piece):

| Column | Meaning |
|---|---|
| `pred_<member>` | the member's prediction, log v85 |
| `contrib_<member>` | its weight × its prediction |
| `pred_stack`, `v85_pred_kmh` | the stacked log v85, and the same in km/h |

A segment's v85 is the mean of its pieces' km/h weighted by `sample_size` + 1
(`ensemble.served`, used in `examples/ensemble.py`).
