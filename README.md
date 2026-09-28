# SafeSpeed: trained models

## What this is

SafeSpeed checks whether a road's posted speed limit is survivable for the people
who use it. It was built for the *AI for Safer Roads 2026* Innovation Challenge
(Asian Development Bank / Agilysis) on the road networks of Maharashtra (India) and
Thailand. This repository holds its three trained models: the code, the trained
weights, sample inputs with their expected results, the sample runs' results and
examples. All three run on an ordinary CPU with Python, numpy and pandas, with no
GPU, internet connection or pickle files.

1. **Design-speed model** (`run_inference.py`, both networks): the speed a road's layout and surroundings invite, what drives it, and the expected harm at that speed. In SafeSpeed it feeds the road priority score, which the methodology report describes; the score itself is not part of this repository.
2. **Vision model** (`run_vision.py`, Thailand): the same from street images alone, plus the survivable speed for the road the images show.
3. **Research ensemble** (`run_ensemble.py`, both networks): the v85 of road pieces from road context, the road network and image features (3 tree models and 4 graph neural networks, combined).

The method, the checks and the full results are in `SafeSpeed_Methodology_Report.pdf`.

## Key terms

- **v50, v85**: the median and 85th-percentile speed of traffic, in km/h.
- **Design speed**: the speed a road's layout and surroundings invite, predicted without seeing measured speeds.
- **Survivable speed**: the highest speed at which a crash is survivable for the person most at risk on the road, for example 30 km/h where people walk beside traffic.
- **Expected harm**: the chance, from 0 to 1, that the person most at risk is killed or seriously injured in a crash at the road's speeds.
- **Segment, road piece**: a SafeSpeed road section, and the shorter TomTom pieces it is made of.
- **POI**: point of interest, such as a school, a bus stop or a shop.
- **iRAP, ThaiRAP**: the International Road Assessment Programme and its Thai programme, whose surveyors code road attributes from images. ThaiRAP's coded survey images trained the image classifiers.
- **Embedding, probe**: the vector of numbers an image model (CLIP or DINOv3) makes from a photo, and a small classifier that reads one road attribute from it.
- **PCA**: principal component analysis. The vision model keeps the first 32 principal components of the image embeddings.
- **Stack**: a weighted sum of several models' predictions.
- **R²**: the share of the variation in log speed that a model explains; 1 is perfect.
- **95 % CI**: the 95 % confidence interval, from resampling whole 50 km blocks of road.
- **Placebo**: the same model retrained with each segment's image features swapped for another segment's. A real gain from the images must beat it.

## Run it

Clone the repository (weights and sample data included), install numpy and pandas and
run the three checks:

```bash
git clone https://github.com/Shishuii/safe_speed.git
cd safe_speed
python3 -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt    # numpy and pandas

python run_inference.py --check
python run_vision.py --check
python run_ensemble.py --check
```

| Command | Runs | Writes |
|---|---|---|
| `run_inference.py` | design-speed model on 200 road segments | `outputs/predictions.csv` |
| `run_vision.py` | vision model on 3 Thai segments (7 street images) | `outputs/vision_predictions.csv` (per segment), `outputs/vision_images.csv` (per image) |
| `run_ensemble.py` | ensemble on 9 road pieces of 5 segments, with their road graph | `outputs/ensemble_predictions.csv` |

`--check` compares every result with the expected results in `data/`, which the full
SafeSpeed system produced for the same rows, and prints `check: PASSED` when they match
(float32 parts within a printed tolerance). `outputs/` already holds these results, and
the commands above rewrite them byte for byte. Tested on Python 3.9 to 3.14,
numpy 1.22 to 2.5 and pandas 1.4 to 3.0.

### Options

The ensemble runs 7 members by default. Two other stacks are included, and
`--vision-from-images` checks that models 2 and 3 chain:

```bash
python run_ensemble.py --check --stack with_dinov3      # 8 members: adds the DINOv3 ThaiRAP classifier
python run_ensemble.py --check --stack without_images   # 4 members: no image features
python run_ensemble.py --check --vision-from-images     # rebuild the image features with model 2 first
```

`python run_vision.py --check --with-dinov3` adds the DINOv3 classifier's `a_*`
columns to model 2; only the `with_dinov3` stack uses them. Runs with `--with-dinov3` or
another `--stack` write that variant's results to `outputs/`; the three plain commands
above restore the shipped files.

Three short, commented examples call the models from Python:

```bash
python examples/design_speed.py    # writes outputs/examples/design_speed.csv
python examples/vision.py          # writes outputs/examples/vision.csv
python examples/ensemble.py        # writes outputs/examples/ensemble_segments.csv
```

New street images have to be embedded first with `--embed-to`, which needs the
packages in `requirements-vision.txt` and downloads the CLIP backbone once
(`--with-dinov3` also downloads the DINOv3 backbone):

```bash
python -m pip install -r requirements-vision.txt
python run_vision.py --images my_images.csv --segments my_segments.csv --image-dir my_jpgs/ \
    --embed-to my_embeddings.npz --detections my_detections.jsonl --output my_vision.csv
```

## Use your own data

Each model reads the same kind of files as its sample, so the samples serve as
templates. `COLUMNS.md` explains every input and output column.

- Model 1: one row per road segment, as in `data/sample_segments.csv`. Run
  `python run_inference.py --input my_roads.csv --output my_predictions.csv`.
- Model 2: images, segments, Mapillary detections and embeddings, as in
  `data/vision_sample/` (`python run_vision.py --help` lists the options).
- Model 3: road pieces and the two edge lists that join them, as in
  `data/graph_sample/`. Run `python run_ensemble.py --graph my_folder/ --output my_ensemble.csv`.
  The ensemble scores prepared graphs only: the network features (`c2g_*`, made with
  city2graph) and the two edge lists are made by SafeSpeed's build code, which is not
  in this repository.

Inputs must be made the same way as the training data: nearby places counted from the
challenge's point-of-interest data, WorldPop 2025 population within 300 m of the road,
and geometry measured on the road lines. Only roads in Maharashtra and Thailand can be
scored (Thailand only for the vision model), and bad rows are refused with a list of
every problem. Do not re-save the CSVs in Excel: the trees split on exact values, so a
rounded input can move a speed by a few km/h.

## What is in the repository

```
safe_speed/                            the GitHub repository
├── README.md, COLUMNS.md              this guide; every input and output column
├── LICENSE.txt                        licence statement for the code and weights
├── SafeSpeed_Methodology_Report.pdf   method, checks and results
├── requirements.txt                   numpy, pandas (requirements-vision.txt: to embed new images)
├── run_inference.py, run_vision.py, run_ensemble.py
├── examples/                          three short Python examples
├── safespeed/                         model code (numpy and pandas)
├── weights/                           trained weights
│   ├── design_speed_trees.json        model 1: 4 tree ensembles
│   ├── injury_curves.json             injury-risk curves (models 1 and 2)
│   ├── vision/                        model 2: probes, PCA, speed trees, rules
│   ├── vision_thairap/                ThaiRAP-trained probes and DINOv3 classifier, with their licences
│   └── ensemble/                      model 3: 8 members, stacks, input rules, MODEL_CARD.md
├── data/                              sample inputs and expected results
│   ├── sample_segments.csv            model 1 sample input
│   ├── expected_predictions.csv       its expected results
│   ├── vision_sample/                 7 Mapillary images, detections, embeddings, expected results
│   └── graph_sample/                  206 road pieces, two edge lists, expected results
└── outputs/                           results of the three runs above (the examples write to outputs/examples/)
```

`data/` holds samples only. The inputs and results for the full networks are built
from the challenge's TomTom data, which is not included.

## Accuracy, limits and method

Scores from five-fold cross-validation on 50 km blocks of road the model did not
see in training (R² of log speed; 1 would be perfect):

| Model | Predicts | R² | R² Maharashtra | R² Thailand | Mean absolute error |
|---|---|---|---|---|---|
| Design speed | v85 of 15,143 segments | 0.607 | 0.416 | 0.512 | 10.6 km/h |
| Vision | v85 of 6,268 Thai segments with images | 0.437 | – | 0.437 | 11.3 km/h |
| Ensemble, default stack | v85 of 46,548 road pieces | 0.669 | 0.490 | 0.661 | – |

The pooled R² also counts the difference between the two countries, so the
per-country values are the fairer measure. On Thai road pieces with images, the image
features change the ensemble's R² by +0.034 (95 % CI +0.022 to +0.044). The DINOv3
classifier member adds a small gain, +0.0010 (+0.0003 to +0.0017), that a placebo matches: with each
segment's classifier scores swapped for another's, the stack does as well (real minus
placebo -0.0011, -0.0027 to +0.0008). None of the gain is credited to the classifier, so
that member is an option and not the default.

Limits:

- A design speed is what a road's layout invites, not what drivers do; its typical error is 10–11 km/h.
- The injury-risk curves come from German crash data and are applied to Indian and Thai roads.
- The shipped ensemble weights are fitted on all the data, so their predictions on the training roads are in-sample; the cross-validated scores above are the accuracy estimate.
- Only roads in Maharashtra and Thailand can be scored, and the vision model covers Thailand only.

The method, the checks and the results are set out in `SafeSpeed_Methodology_Report.pdf`;
the ensemble has its own model card, `weights/ensemble/MODEL_CARD.md`.

## Licences and attribution

- **Code and weights**: the SafeSpeed team's submission to the AI for Safer Roads 2026 Innovation Challenge, provided as is, without warranty (`LICENSE.txt`).
- **ThaiRAP-trained parts are non-commercial.** The probes and the DINOv3 classifier in `weights/vision_thairap/` are CC BY-NC 4.0, and so is everything trained on their outputs: the vision speed model, the ensemble's image members and its `default` and `with_dinov3` stacks. That covers any vision results you compute, and ensemble results from the `default` or `with_dinov3` stack. Model 1 and the `without_images` stack do not use them.
- **ThaiRAP training data**: the images are from UCL's data record doi:10.5522/04/26520787.v1 (CC BY-NC 4.0) and the labels from the V-RoAst repository (github.com/PongNJ/V-RoAst). Neither is included (`weights/vision_thairap/LICENSE.txt`).
- **DINOv3**: the classifier runs on features of Meta's DINOv3. The classifier heads and the cached DINOv3 features of the sample images are distributed under the DINOv3 License (copy in `weights/vision_thairap/DINOv3_LICENSE.md`), and the heads also carry CC BY-NC 4.0. Only the `a_*` columns and the `with_dinov3` stack use them. The DINOv3 backbone is not included.
- **Mapillary** images and detections: © Mapillary contributors, CC BY-SA 4.0 (`data/vision_sample/ATTRIBUTION.md`).
- **OpenStreetMap** road data and tags: © OpenStreetMap contributors, ODbL, with Overture Maps Foundation data.
- **TomTom** speeds and posted limits, and the POI counts: challenge data, used under the terms of the AI for Safer Roads 2026 Innovation Challenge; the raw data is not included.
- **WorldPop** R2025A population: CC BY 4.0. **CLIP** (OpenAI, MIT licence) is downloaded only to embed new images.
- Injury-risk curves: Lubbe, N., Wu, Y. & Jeppsson, H. (2022). Traffic Safety Research 2:000006, doi:10.55329/vfma7555.
- V-RoAst: Jongwiriyanurak, N., Zeng, Z., Wang, M., Haworth, J., Tanaksaranond, G. & Boehm, J. (2025). V-RoAst: Visual Road Assessment. ICCV Workshops, 1669–1678, doi:10.1109/ICCVW69036.2025.00176.

The code and this guide were written with AI assistance (Anthropic Claude).
