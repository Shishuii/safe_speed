"""Where the code and the data and weights are published, and a check that the
data and weights are in place.

The code lives in the GitHub repository. The weights and sample data are one
separate download, DATA_ZIP, from the Google Drive folder DATA_URL; unzipped in
the repository folder, it puts data/ and weights/ next to run_inference.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

GITHUB_URL = "https://github.com/Shishuii/safe_speed"
DATA_URL = "https://drive.google.com/drive/folders/1j-Lf5c8Cs0SkGGhmWzDX-TC30KZAR8ni"
DATA_ZIP = "SafeSpeed_data_weights.zip"
FOLDERS = ("data", "weights")


def missing_assets(here) -> list:
    """The folders of FOLDERS that are not next to the run scripts in `here`."""
    return [name for name in FOLDERS if not (Path(here) / name).is_dir()]


def require_assets(here) -> None:
    """Stop with exit code 2 and plain instructions when data/ or weights/ is missing.

    `here` is the folder that holds run_inference.py, run_vision.py and
    run_ensemble.py.
    """
    here = Path(here).resolve()
    missing = missing_assets(here)
    if not missing:
        return
    lines = [
        "The data and weights are not in this folder yet "
        f"({' and '.join(m + '/' for m in missing)} not found).",
        f"Download {DATA_ZIP} from",
        f"  {DATA_URL}",
        "and unzip it here, so that data/ and weights/ sit next to run_inference.py:",
        f"  {here}",
    ]
    nested = here / Path(DATA_ZIP).stem
    if any((nested / name).is_dir() for name in FOLDERS):
        lines.append(f"They were unzipped into {nested.name}/: move data/ and weights/ "
                     "up one level, out of that folder.")
    print("\n".join(lines), file=sys.stderr)
    sys.exit(2)
