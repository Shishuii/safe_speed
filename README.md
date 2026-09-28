# SafeSpeed: trained models

## What this is

SafeSpeed checks whether a road's posted speed limit is survivable for the people
who use it. It was built for the *AI for Safer Roads 2026* Innovation Challenge
(Asian Development Bank / Agilysis) on the road networks of Maharashtra (India) and
Thailand. This repository holds the code, examples and results of its three trained
models; their weights and sample inputs are a separate download from Google Drive.
All three run on an ordinary CPU with Python, numpy and pandas, with no GPU,
internet connection or pickle files.

1. **Design-speed model** (`run_inference.py`, both networks): the speed a road's layout and surroundings invite, what drives it, and the expected harm at that speed. It feeds the SafeSpeed priority score.
2. **Vision model** (`run_vision.py`, Thailand): the same from street images alone, plus the survivable speed for the road the images show.
3. **Research ensemble** (`run_ensemble.py`, both networks): the v85 of road pieces from road context, the road network and image features (3 tree models and 4 graph neural networks, combined).

## Key terms

- **v50, v85**: the median and 85th-percentile speed of traffic, in km/h.
- **Design speed**: the speed a road's layout and surroundings invite, predicted without seeing measured speeds.
- **Survivable speed**: the highest speed at which a crash is survivable for the person most at risk on the road, for example 30 km/h where people walk beside traffic.
- **Expected harm**: the chance, from 0 to 1, that the person most at risk is killed or seriously injured in a crash at the road's speeds.
- **Segment, road piece**: a SafeSpeed road section, and the shorter TomTom pieces it is made of.
- **Embedding, probe**: the vector of numbers an image model (CLIP or DINOv3) makes from a photo, and a small classifier that reads one road attribute from it.
- **Stack**: a weighted sum of several models' predictions.
- **ThaiRAP**: the Thai Road Assessment Programme, whose coded survey images trained the image classifiers.

## Run it

Get the code:

```bash
git clone https://github.com/Shishuii/safe_speed.git
cd safe_speed
```

Get the data and weights: download **`SafeSpeed_data_weights.zip`** (about 12 MB unzipped) from the
[Google Drive folder](https://drive.google.com/drive/folders/1j-Lf5c8Cs0SkGGhmWzDX-TC30KZAR8ni) and unzip it in the `safe_speed` folder, so that
`data/` and `weights/` sit next to `run_inference.py`. Then install numpy and pandas
and run the three checks:

```bash
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

`--check` compares every result with the full SafeSpeed pipeline's results for the
same rows and prints `check: PASSED` when they match (float32 parts within a printed
tolerance). `outputs/` already holds these results. Tested on Python 3.9 to 3.14,
numpy 1.22 to 2.5 and pandas 1.4 to 3.0.

The ensemble runs 7 members by default. Two other stacks are included, and
`--vision-from-images` checks that models 2 and 3 chain:

```bash
python run_ensemble.py --check --stack with_dinov3      # 8 members: adds the DINOv3 ThaiRAP classifier
python run_ensemble.py --check --stack without_images   # 4 members: no image features
python run_ensemble.py --check --vision-from-images     # rebuild the image features with model 2 first
```

`python run_vision.py --check --with-dinov3` adds the DINOv3 classifier's `a_*`
columns to model 2; only the `with_dinov3` stack uses them.

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
  The package scores prepared graphs only: the city2graph features and edge lists
  for new roads come from the SafeSpeed build pipeline, which is not included.

Only roads in Maharashtra and Thailand can be scored (Thailand only for the vision
model), and bad rows are refused with a list of every problem. Do not re-save the CSVs in
Excel: the trees split on exact values, so a rounded input can move a speed by a few
km/h. Inputs must be made the way the SafeSpeed pipeline made them: nearby places
from the challenge's feature files, WorldPop population within 300 m and geometry
from the road lines.

## What is in the folder

```
safe_speed/                            the GitHub repository, about 3 MB
├── README.md, COLUMNS.md              this guide; every input and output column
├── LICENSE.txt                        terms for the code and weights
├── SafeSpeed_Methodology_Report.pdf   method, checks and results
├── requirements.txt                   numpy, pandas (requirements-vision.txt: to embed new images)
├── run_inference.py, run_vision.py, run_ensemble.py
├── examples/                          three short Python examples
├── safespeed/                         model code (numpy and pandas)
├── weights/                           from the Google Drive download
│   ├── design_speed_trees.json        model 1: 4 tree ensembles
│   ├── injury_curves.json             injury-risk curves (models 1 and 2)
│   ├── vision/                        model 2: probes, PCA, speed trees, rules
│   ├── vision_thairap/                ThaiRAP-trained probes and DINOv3 classifier, with their licences
│   └── ensemble/                      model 3: 8 members, stacks, input rules, MODEL_CARD.md
├── data/                              from the Google Drive download
│   ├── sample_segments.csv            model 1 sample input
│   ├── expected_predictions.csv       its expected results
│   ├── vision_sample/                 7 Mapillary images, detections, embeddings, expected results
│   └── graph_sample/                  206 road pieces, two edge lists, expected results
└── outputs/                           results of the runs above; examples/ holds the examples' results
```

`data/` holds samples only. The inputs and results for the full networks are built
from the challenge's TomTom data and are not included.

## Accuracy and method

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

A design speed is what the layout invites, not what drivers do (typical error
10–11 km/h). The injury curves come from German crash data. The shipped ensemble
weights are fitted on all the data, so they are in-sample on the training roads.
Details: `SafeSpeed_Methodology_Report.pdf` and `weights/ensemble/MODEL_CARD.md`.

## Licences

- **Code and other weights**: no licence chosen yet; shared for evaluation. Ask the SafeSpeed team before reusing them (`LICENSE.txt`).
- **ThaiRAP-trained parts are non-commercial.** The probes and the DINOv3 classifier in `weights/vision_thairap/` are CC BY-NC 4.0, and so is everything trained on their outputs: the vision speed model, the ensemble's image members and its `default` and `with_dinov3` stacks. That covers any vision results you compute, and ensemble results from the `default` or `with_dinov3` stack. Model 1 and the `without_images` stack do not use them.
- **ThaiRAP labels.** The images come from UCL's data record doi:10.5522/04/26520787.v1 (CC BY-NC 4.0), but the labels came from the V-RoAst repository, which declares no licence. Get the V-RoAst authors' written confirmation that the labels fall under the same terms before you pass the image weights on (`weights/vision_thairap/LICENSE.txt`).
- **DINOv3**: the classifier runs on features of Meta's DINOv3, under the DINOv3 License (copy in `weights/vision_thairap/DINOv3_LICENSE.md`). Only the `a_*` columns and the `with_dinov3` stack use it. Whether the classifier heads and the cached sample features are derivative works of DINOv3 is not settled. If they are, the DINOv3 License applies to them alongside CC BY-NC 4.0: settle this before passing them on.
- **Mapillary** images and detections: © Mapillary contributors, CC BY-SA 4.0 (`data/vision_sample/ATTRIBUTION.md`).
- **OpenStreetMap** road data and tags: © OpenStreetMap contributors, ODbL, with Overture Maps Foundation data.
- **TomTom** speeds and posted limits, and the POI counts: AI for Safer Roads 2026 challenge dataset terms.
- **WorldPop** R2025A population: CC BY 4.0. **CLIP** (OpenAI, MIT) is downloaded only to embed new images.
- Injury-risk curves: Lubbe, N., Wu, Y. & Jeppsson, H. (2022). Traffic Safety Research 2:000006, doi:10.55329/vfma7555.

The code and this guide were written with AI assistance (Anthropic Claude).
