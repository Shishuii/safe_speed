# SafeSpeed research ensemble (v85): model card

Predicts the 85th-percentile operating speed (v85, km/h) of a road piece (a short TomTom piece of a SafeSpeed segment) from road context, its position in the road graph and, in Thailand, its street images. Research model: it runs beside the design-speed model and is not part of the SafeSpeed priority score. Run it with `python run_ensemble.py --check`. In the graph, each road piece is a node.

## Members and stacks

8 members are shipped; each predicts log(v85). A stack is a non-negative linear combination (NNLS, weights and intercept >= 0): `log v85 = intercept + sum(weight[m] * member[m])`, then `exp`. A SafeSpeed segment (`old_id`) gets the (sample_size + 1)-weighted mean of its road pieces' km/h.

| Member | Kind | Graph | Inputs | `default` | `with_dinov3` | `without_images` |
|---|---|---|---|---|---|---|
| `gbm_base` | gradient-boosted trees (400, depth 6) | - | 15 | 0.1091 | 0.1061 | 0.1832 |
| `gbm_c2g` | gradient-boosted trees (400, depth 6) | - | 28 | 0.1913 | 0.1652 | 0.4504 |
| `gnn_topological` | GraphSAGE (2 x SAGEConv 96, skip, head 48) | topological | 28 | 0.0923 | 0.0947 | 0.2497 |
| `gnn_c2g_knn` | GraphSAGE (2 x SAGEConv 96, skip, head 48) | knn | 28 | 0.0176 | 0.0182 | 0.1162 |
| `gbm_c2g_vision` | gradient-boosted trees (400, depth 6) | - | 71 | 0.3119 | 0.2188 | - |
| `gnn_topological_vision` | GraphSAGE (2 x SAGEConv 96, skip, head 48) | topological | 71 | 0.2126 | 0.2090 | - |
| `gnn_c2g_knn_vision` | GraphSAGE (2 x SAGEConv 96, skip, head 48) | knn | 71 | 0.0612 | 0.0595 | - |
| `gbm_c2g_vision_irap` | gradient-boosted trees (400, depth 6) | - | 141 | - | 0.1251 | - |
| intercept | | | | 0.0159 | 0.0144 | 0.0000 |

`default` uses 7 members: all but `gbm_c2g_vision_irap`, the DINOv3 classifier member. `with_dinov3` adds that member (8 members). On the Thai road pieces with images it gives a small stack gain, +0.0010 (95% CI +0.0003 to +0.0017), that a placebo with shuffled classifier scores matches (real minus placebo -0.0011), so none of it is credited to the classifier (see below). `without_images` keeps only the members with no image features. Pick one with `python run_ensemble.py --stack <name>`. Each stack was fitted on its own members' out-of-fold predictions, so its weights differ. `per_fold_weights_oof_diagnostic` in `stack.json` holds each cross-validation fold's weights for the `default` stack; they are diagnostics of the out-of-fold evaluation, not for inference.

Graphs. `topological`: road pieces whose end points lie within 1 m of each other are joined; a piece left with no neighbour is joined to up to 8 nearest piece midpoints within 1,500 m; 1,815 pieces still have no neighbour (their neighbour mean is 0). `knn`: city2graph `knn_graph` (k = 12) over piece midpoints in UTM, including the duplicate self-loops it produces where midpoints coincide. Both graphs are symmetric (every edge stored in both directions), and the package assumes that. Edges are read as `src -> dst`; a GNN output depends on the node's 2-hop in-neighbourhood only.

**New roads.** The package scores prepared graphs; it does not build them. The 13 `c2g_*` features and both edge lists were made by SafeSpeed's build code (city2graph 0.4.0 and geopandas), which is not part of this package; a new network needs them made the same way. The `v_*` and `a_*` features come from the vision model (`run_vision.py`). `inputs.json` holds the rules every `nodes.csv` is checked against.

## Features

Raw inputs, in the order of `features.json`. Trees take them raw (blank = NaN allowed). GraphSAGE members fill a blank with the column's frozen median, z-score with the frozen mean and sd (float64; 0 where sd is 0), then cast to float32; the statistics are in each `gnn_*.json` (computed on all 46,548 road pieces, features only).

**`gbm_base`** (15):

- `posted_limit`: posted speed limit, km/h (TomTom)
- `frc`: TomTom functional road class 0-7 (0 = motorway)
- `length_log`: log(1 + segment length in m)
- `poi_vru_log`: log(1 + weighted count of places that bring people on foot)
- `pop_log`: log(1 + residents within 300 m, WorldPop 2025)
- `pop_density_log`: log(1 + mean residents per 100 m cell within 300 m)
- `school_log`: log(1 + schools nearby)
- `transport_log`: log(1 + public-transport stops nearby)
- `geo_sinuosity`: road length / straight-line length
- `geo_curvature`: total turning angle per metre, rad/m
- `geo_turn_sharp`: count of sharp (> 45 deg) direction changes
- `geo_junction_density`: other segment ends within 35 m of this segment's ends, per km
- `lat`: latitude of the segment (WGS-84)
- `lon`: longitude of the segment (WGS-84)
- `country_TH`: 1 for Thailand, 0 for Maharashtra

**added in `gbm_c2g`, `gnn_topological`, `gnn_c2g_knn`** (13):

- `c2g_dual_degree`: city2graph dual (line) graph: segments meeting this one at a junction
- `c2g_dual_clustering`: dual graph clustering coefficient
- `c2g_pagerank`: dual graph PageRank
- `c2g_betweenness`: dual graph betweenness (800-pivot approximation): through-route function
- `c2g_closeness_local`: dual degree / 2-hop ego size
- `c2g_ego_size`: segments within 2 hops on the dual graph
- `c2g_knn_dist_mean`: mean distance (m) to the 8 nearest segment midpoints
- `c2g_radius_count`: segment midpoints within 500 m
- `c2g_emst_degree`: degree in the Euclidean minimum spanning tree of midpoints
- `c2g_delaunay_degree`: degree in the Delaunay graph of midpoints
- `c2g_gabriel_degree`: degree in the Gabriel graph of midpoints
- `c2g_rng_degree`: degree in the relative-neighbourhood graph of midpoints
- `c2g_planar_ratio`: RNG degree / Delaunay degree (how cluttered the local pattern is)

**added in the three `*_vision` members** (43):

- `v_lanes`: expected lane count from a CLIP probe trained on OSM lane tags (image-derived)
- `v_shoulder`: paved shoulder width class 0-3 (ThaiRAP probe), mean over the images
- `v_roadside_dist`: distance to roadside hazard class 0-3 (ThaiRAP probe)
- `v_roadside_severe`: share of images with a severe roadside object (ThaiRAP probe)
- `v_roadside_barrier`: share of images with a roadside safety barrier (ThaiRAP probe)
- `v_skid`: skid resistance / surface class 0-4 (ThaiRAP probe)
- `v_lighting`: share of images with street lighting (ThaiRAP probe)
- `v_calming`: share of images with traffic calming (ThaiRAP probe)
- `v_access`: property access intensity class 0-3 (ThaiRAP probe)
- `v_service_road`: share of images with a service road (ThaiRAP probe)
- `has_vision`: 1 if the served segment has usable street images, else 0
- `v_emb_00` ... `v_emb_31`: the segment's mean CLIP ViT-L/14 image embedding on 32 principal components

**added in `gbm_c2g_vision_irap`** (70):

- `a_<attribute>__<iRAP code>`: the DINOv3 ThaiRAP classifier's probability of each iRAP code, averaged over the segment's images, for its 19 design-role attributes:
  - `a_land_use_ds__*` (Land use - driver-side): codes 1, 3, 4, 6, 7
  - `a_land_use_ps__*` (Land use - passenger-side): codes 1, 3, 4, 6, 7
  - `a_roadside_dist_ds__*` (Roadside severity - driver-side distance): codes 1, 2, 3, 4
  - `a_roadside_obj_ds__*` (Roadside severity - driver-side object): codes 1, 2, 3, 8, 9, 11, 12, 13, 15, 17
  - `a_roadside_dist_ps__*` (Roadside severity - passenger-side distance): codes 1, 2, 3, 4
  - `a_roadside_obj_ps__*` (Roadside severity - passenger-side object): codes 1, 2, 3, 8, 9, 11, 12, 13, 15, 17
  - `a_shoulder_rumble__*` (Shoulder rumble strips): codes 1, 2
  - `a_paved_shoulder_ds__*` (Paved shoulder - driver-side): codes 1, 2, 3, 4
  - `a_paved_shoulder_ps__*` (Paved shoulder - passenger-side): codes 1, 2, 3, 4
  - `a_property_access__*` (Property access points): codes 1, 3, 4
  - `a_lanes__*` (Number of lanes): codes 2, 3, 4
  - `a_curvature__*` (Curvature): codes 1, 2
  - `a_skid_resistance__*` (Skid resistance / grip): codes 1, 2
  - `a_delineation__*` (Delineation): codes 1, 2
  - `a_street_lighting__*` (Street lighting): codes 1, 2
  - `a_ped_fencing__*` (Pedestrian fencing): codes 1, 2
  - `a_traffic_calming__*` (Speed management / traffic calming): codes 1, 2
  - `a_parking__*` (Vehicle parking): codes 1, 2
  - `a_service_road__*` (Service road): codes 1, 2

Image values are per SafeSpeed segment, copied to its road pieces; Maharashtra and Thai segments without usable images have them blank and `has_vision = 0`. They are outputs of the vision model in this package: `run_ensemble.py --vision-from-images` recomputes them from the sample images and finds the attribute columns and `has_vision` identical and the 32 float32 embedding components equal to float32 precision, with unchanged predictions; with `--stack with_dinov3` it also recomputes the classifier scores, which match to float32 precision too. `v_lanes` is the expected lane count from a CLIP probe trained on OpenStreetMap lane tags, cross-fitted by 50 km block: an image-derived design attribute. Observed OSM lanes are never an input, and refitting the tree image member without `v_lanes` leaves its image gain unchanged (below).

## The DINOv3 classifier member

`gbm_c2g_vision_irap` is `gbm_c2g_vision` plus the class probabilities of a ThaiRAP classifier. The classifier (`weights/vision_thairap/dinov3_heads.npz`, 30 attributes, 109 classes) is one linear softmax head per ThaiRAP attribute on frozen DINOv3 ViT-L/16 features (5,120 values per image, pooled views of the image tokens; see `safespeed/vision/classifier.py`), trained on the ThaiRAP survey images and labels with mirrored images (driver/passenger-side labels swapped). The member uses only its 19 design-role attributes; the 11 reference-role heads (who uses the road, junctions, median, area type) never enter a model.

On a ThaiRAP holdout (every 5th 2 km route block held out; 119 segments), with V-RoAst's metric over the 30 attributes the classifier is trained on: accuracy 0.840, F1 0.530. Gemini 1.5 Flash, zero-shot, on the same segments and attributes: 0.709, 0.363. Mean kappa over the 25 attributes with more than one code on the holdout: 0.340 (16 of them reliable); Gemini 0.107. V-RoAst's own averages run over 39 attributes, 9 of which have a single code in all of ThaiRAP. The classifier has no head for those and returns that code, which lifts its averages to accuracy 0.877 and F1 0.639 (Gemini 0.775 and 0.484).

Out-of-fold, on the 11,314 Thai road pieces with images (199 spatial blocks, 1000 block-bootstrap resamples):

- `with_dinov3` over `default`: +0.0010 (95% CI +0.0003 to +0.0017); Thai road pieces without images -0.0001 (95% CI -0.0005 to +0.0002); Maharashtra -0.0003 (95% CI -0.0012 to +0.0008)
- `gbm_c2g_vision_irap` over `gbm_c2g_vision`: +0.0000 (95% CI -0.0066 to +0.0050)
- placebo (each covered segment given another segment's classifier scores, member retrained, stack refitted): member R² 0.6709, stack R² 0.6739; real over placebo: stack -0.0011 (95% CI -0.0027 to +0.0008), member -0.0027 (95% CI -0.0076 to +0.0030)
- without the 78 segments that straddle folds or share an image (383 road pieces): stack gain +0.0009 (95% CI +0.0003 to +0.0016)

The classifier's scores add no measurable information: the member is no better than `gbm_c2g_vision`, and the placebo does as well as the real scores, so the small stack gain comes from having a second image tree model, not from what the classifier sees. The image features already include CLIP probes trained on the same ThaiRAP labels. The member is therefore left out of the `default` stack and shipped as the `with_dinov3` option.

## Training protocol

Target: log(v85) on the 46,548 road pieces with a valid speed profile and at least 30 probe passes. Folds: 5, made of 50 km blocks with each named street and carriageway pair kept whole. Rows are weighted by reliability n/(n+200), mean 1 per country. Trees: scikit-learn HistGradientBoostingRegressor (max_depth 6, learning rate 0.06, 400 iterations, L2 1.0, seed 42). GraphSAGE: 400 epochs, Adam (learning rate 0.01, weight decay 0.0005), dropout 0.2, weighted Huber (delta 1.0), seed 42, deterministic CUDA. Stack: per-fold NNLS (weights and intercept >= 0). The image members, and the DINOv3 member, use the same folds, weights and hyper-parameters. Gains are measured on identical rows with 1,000 resamples of whole 50 km blocks.

## Out-of-fold accuracy (R², log v85)

| Model | all | Maharashtra | Thailand | TH with images | TH without images | Spearman |
|---|---|---|---|---|---|---|
| `gbm_base` | 0.6454 | 0.4599 | 0.6217 | 0.6245 | 0.6134 | 0.8291 |
| `gbm_c2g` | 0.6591 | 0.4788 | 0.6411 | 0.6337 | 0.6439 | 0.8346 |
| `gnn_topological` | 0.6195 | 0.4490 | 0.5319 | 0.5370 | 0.5199 | 0.8181 |
| `gnn_c2g_knn` | 0.5633 | 0.3628 | 0.4732 | 0.4564 | 0.4836 | 0.7776 |
| `gbm_c2g_vision` | 0.6630 | 0.4793 | 0.6572 | 0.6681 | 0.6406 | 0.8376 |
| `gnn_topological_vision` | 0.6402 | 0.4639 | 0.5906 | 0.5913 | 0.5840 | 0.8260 |
| `gnn_c2g_knn_vision` | 0.6059 | 0.4082 | 0.5616 | 0.5609 | 0.5561 | 0.8107 |
| `gbm_c2g_vision_irap` | 0.6614 | 0.4761 | 0.6567 | 0.6682 | 0.6396 | 0.8366 |
| placebo `gnn_topological_vision` (image features shuffled between segments) | 0.5923 | 0.4305 | 0.4529 | 0.3626 | 0.5408 | 0.8045 |
| placebo `gnn_c2g_knn_vision` (image features shuffled between segments) | 0.6147 | 0.4402 | 0.5302 | 0.5011 | 0.5544 | 0.8132 |
| **`without_images` stack (4 members)** | 0.6653 | 0.4910 | 0.6414 | 0.6376 | 0.6403 | 0.8384 |
| **`default` stack (7 members)** | 0.6691 | 0.4897 | 0.6608 | 0.6718 | 0.6444 | 0.8410 |
| **`with_dinov3` stack (8 members)** | 0.6690 | 0.4894 | 0.6613 | 0.6728 | 0.6442 | 0.8410 |
| exploratory: 3 vision members + gbm_base (see note) | 0.6698 | 0.4893 | 0.6651 | 0.6808 | 0.6438 | 0.8417 |

The image gain that survives the placebo is carried by `gbm_c2g_vision`. The GNN vision members' gains are not distinguishable from their placebos (one placebo still beats its no-vision member), so they are not attributed to the images. Exploratory stack: 3 vision members + gbm_base, chosen after seeing the 7-member weights; reported for comparison only.

On the 11,314 Thai road pieces with images (identical rows for both, 199 spatial blocks, 1000 block-bootstrap resamples):

- `default` over `without_images`: +0.0342 (95% CI +0.0221 to +0.0444)
- `gbm_c2g_vision` over `gbm_c2g`: +0.0344 (95% CI +0.0233 to +0.0445)
- placebo (each segment given another segment's image features, all members retrained and restacked): stack R² 0.6372; real over placebo +0.0346 (95% CI +0.0230 to +0.0449). GBM placebo gains: node permutation -0.0055, served-segment permutation -0.0159
- without the 78 segments that straddle folds or share an image with a segment in another fold (383 road pieces): +0.0333 (95% CI +0.0219 to +0.0441)
- `gbm_c2g_vision` refitted without `v_lanes`: gain over `gbm_c2g` +0.0344 (95% CI +0.0227 to +0.0443)
- elsewhere: Thai road pieces without images +0.0040 (95% CI -0.0002 to +0.0084); Maharashtra -0.0013 (95% CI -0.0070 to +0.0036)

## Caveats

- **The shipped weights are in-sample.** Every member here is refitted on all 46,548 labelled road pieces, and the stack weights are one NNLS fit on all rows' out-of-fold predictions. Predictions from these files (and `expected.csv`) are in-sample and do not reproduce the out-of-fold numbers above, which come from 5-fold models that never saw the held-out block. Scored on the road pieces they were fitted on, the shipped files give R² 0.7801 (`default`), 0.7810 (`with_dinov3`) and 0.7793 (`without_images`): a fit statistic, not an accuracy estimate. (The all-data `default` weights applied to the out-of-fold member predictions give 0.6753.)
- **GBM early stopping.** The GBM members use scikit-learn's default `early_stopping='auto'`, which on more than 10,000 rows holds out a random 10 % internally as a validation split (random_state 42). It never stopped early (all 400 iterations ran), but each fitted GBM saw 90 % of the rows.
- **GNN sensitivity.** The GraphSAGE members react to tiny input changes. Z-scoring the 32 embedding components in float32 (their stored type) instead of float64 moves `gnn_topological_vision` from 0.6402 to 0.6389, `gnn_c2g_knn_vision` from 0.6059 to 0.6299 and the `default` stack from 0.6691 to 0.6733; the tree members do not change, and the GraphSAGE placebos move by several points in both directions. Read GraphSAGE differences of about 0.02 as training noise. The robust results are the tree image gain (+0.0344) and the placebo-controlled stack gain on Thai road pieces with images (+0.0346).
- The GraphSAGE members are re-stated in numpy. They match torch up to float32 summation order, so they are checked to 1e-05 relative; the trees exactly.
- `expected.csv` rows at level `served` give `v85_measured` as the same weighted mean of the children's measured v85.
- The sample (206 nodes, 9 targets) is the 2-hop in-neighbourhood of the targets in both graphs; only target nodes carry full-graph predictions.
- The per-fold stack weights are fitted on out-of-fold member predictions from the other folds, whose models saw the held-out fold's labels: the usual optimism of cross-validated stacking, small with at most 8 weights. The reported R² is that of the per-fold stack.

## Provenance

Trained with SafeSpeed's training code on an NVIDIA GeForce RTX 2080 Ti. The code is not included; the sha256 of each training script (`train_ensemble_vision.py`, its cross-validation protocol `cv_protocol_script` and the classifier's `train_irap_classifier.py`) and of the classifier's inputs is recorded under `provenance` in every member JSON and `stack.json`. The DINOv3 classifier heads, `dinov3_heads.npz`, have sha256 `4a3d070f0cfe7b1b...`.

| Library | Version |
|---|---|
| python | 3.11.9 |
| numpy | 2.2.6 |
| pandas | 3.0.2 |
| scipy | 1.15.3 |
| sklearn | 1.8.0 |
| torch | 2.9.0+cu128 |
| torch.version.cuda | 12.8 |
| torch_geometric | 2.8.0 |
| city2graph | 0.4.0 |
| timm (DINOv3 features) | 1.0.25 |

The inference code needs only numpy and pandas. Training data: TomTom Move speeds (targets), OpenStreetMap / Overture road network, WorldPop 2025, the challenge's POI counts, and Mapillary street images (CC BY-SA 4.0) for the image features.

## Licences of the image members

- **Non-commercial.** Nine of the image features (`v_shoulder`, `v_roadside_dist`, `v_roadside_severe`, `v_roadside_barrier`, `v_skid`, `v_lighting`, `v_calming`, `v_access`, `v_service_road`) are outputs of the ThaiRAP-trained probes, and the `a_*` features of the DINOv3 classifier, which was trained on the same ThaiRAP labels. Both are in `weights/vision_thairap/` and are for non-commercial use only (CC BY-NC 4.0; see the `LICENSE.txt` there). The `*_vision` members, `gbm_c2g_vision_irap`, and the `default` and `with_dinov3` stacks are trained on them and are non-commercial too. The `without_images` stack and its 4 members use neither.
- **DINOv3.** The classifier's features come from Meta's DINOv3 ViT-L/16 (`timm/vit_large_patch16_dinov3.lvd1689m`, revision `30c1109559f6`), licensed under the DINOv3 License (https://ai.meta.com/resources/models-and-libraries/dinov3-license; the full text, which governs, is in `weights/vision_thairap/DINOv3_LICENSE.md`). The backbone weights are not included. The licence has no non-commercial limit, but it sets conditions on the DINO Materials and any derivative works of them: they may only be distributed under the DINOv3 License, with a copy of it (1.b.i); published research must acknowledge DINOv3 (1.b.ii); use must comply with the law, including trade controls (1.b.iii); no reverse engineering, decompiling or discovering their underlying components (1.b.iv); and no use for activities subject to ITAR or end uses that trade controls prohibit, including military or warfare, nuclear, espionage and weapons (1.b.v). The classifier heads are distributed under the DINOv3 License and also carry the CC BY-NC 4.0 limit of the ThaiRAP labels; `gbm_c2g_vision_irap` and the `with_dinov3` stack, which are trained on the heads' outputs, are shipped on the same terms. The `default` stack does not use the classifier.
