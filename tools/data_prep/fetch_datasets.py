"""
Download, extract and organize the six public breast-ultrasound datasets used in
DSMA-Breast (BUSI, BLUI, BrEaST, BUS-BRA, BUS-UCLM, BUS-CoT) into a unified
per-dataset dataset.json schema:

    [{"image": <relative path>, "pathology": "benign|malignant|normal",
      "clinical_data": <free-text descriptors or empty string>}, ...]

Each downloader is idempotent: existing zips are reused, existing extracted
trees are not re-extracted, and dataset.json is regenerated on every run.

Usage:
    python tools/data_prep/fetch_datasets.py                  # all six datasets
    python tools/data_prep/fetch_datasets.py --only BUSI BLUI # subset by name
    python tools/data_prep/fetch_datasets.py --skip-download  # work offline on already-downloaded zips
"""

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR = REPO_ROOT / "datasets"

CHUNK_SIZE = 1 << 20  # 1 MiB
HTTP_TIMEOUT = 60
MAX_RETRIES = 5
USER_AGENT = "DSMA-Breast-Fetcher/1.0"


# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

def download_file(url, output_file, expected_size=None):
    """Stream url to output_file with resume + retry support.

    Skips the download if the on-disk file already matches expected_size.
    Resumes via HTTP Range when the partial file is shorter than expected.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    have = output_file.stat().st_size if output_file.exists() else 0
    if expected_size is not None and have == expected_size:
        return

    headers = {"User-Agent": USER_AGENT}
    if have > 0:
        headers["Range"] = f"bytes={have}-"

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, allow_redirects=True,
                              timeout=HTTP_TIMEOUT, headers=headers) as response:
                if response.status_code == 416:
                    # Range not satisfiable: server says we already have it
                    return
                if response.status_code not in (200, 206):
                    raise IOError(f"HTTP {response.status_code} for {url}")

                total = int(response.headers.get("content-length", 0)) + have
                mode = "ab" if response.status_code == 206 else "wb"
                if mode == "wb":
                    have = 0  # server ignored Range; start over

                with open(output_file, mode) as f, tqdm(
                    total=total if total else None,
                    initial=have,
                    unit="B", unit_scale=True, unit_divisor=1024,
                    desc=f"Downloading {output_file.name}",
                    leave=False,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bar.update(len(chunk))
            return
        except (requests.RequestException, IOError) as exc:
            last_error = exc
            wait = min(60, 2 ** attempt)
            tqdm.write(f"[{output_file.name}] attempt {attempt}/{MAX_RETRIES} failed: {exc}. "
                       f"Retrying in {wait}s...")
            time.sleep(wait)
            have = output_file.stat().st_size if output_file.exists() else 0
            headers["Range"] = f"bytes={have}-"

    raise RuntimeError(f"Failed to download {url} after {MAX_RETRIES} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Zip handling
# ---------------------------------------------------------------------------

def is_valid_zip(zip_path):
    if not zip_path.exists():
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.namelist()
        return True
    except zipfile.BadZipFile:
        return False


def extract_zip(zip_path, output_dir, desired_mapping):
    """Extract files whose path begins with one of desired_mapping's keys.

    The source prefix is rewritten to the target prefix while preserving the
    remaining directory structure. Skips files that are already present.
    """
    output_dir = Path(output_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = []
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            for src_prefix, dst_prefix in desired_mapping.items():
                if name.startswith(src_prefix):
                    rel = Path(name).relative_to(src_prefix)
                    target = output_dir / dst_prefix / rel
                    members.append((name, target))
                    break

        if members and all(t.exists() for _, t in members):
            return

        for name, target in tqdm(members, unit="file",
                                 desc=f"Extracting {zip_path.name}", leave=False):
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as dst:
                while True:
                    buf = src.read(CHUNK_SIZE)
                    if not buf:
                        break
                    dst.write(buf)


# ---------------------------------------------------------------------------
# Per-dataset "organize" functions (build dataset.json)
# ---------------------------------------------------------------------------

def organize_breast(folder):
    xlsx = folder / "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    df = pd.read_excel(xlsx)
    entries = []
    for _, row in df.iterrows():
        clinical = (f"Age:{row['Age']}, Composition:{row['Tissue_composition']}, "
                    f"Symptoms:{row['Symptoms']}, Shape:{row['Shape']}, "
                    f"Echogenicity:{row['Echogenicity']}, Calcifications:{row['Calcifications']}, "
                    f"Thickening:{row['Skin_thickening']}, Signs:{row['Signs']}")
        entries.append({
            "image": f"./images_and_masks/{row['Image_filename']}",
            "pathology": row["Classification"],
            "clinical_data": clinical,
        })
    _write_json(folder / "dataset.json", entries)


def organize_busbra(folder):
    df = pd.read_csv(folder / "bus_data.csv")
    entries = [{
        "image": f"./Images/{row['ID']}.png",
        "pathology": row["Pathology"],
        "clinical_data": "",
    } for _, row in df.iterrows()]
    _write_json(folder / "dataset.json", entries)


def organize_busuclm(folder):
    # Label is inferred from the colored overlay in the BUS-UCLM mask
    def label_from_mask(mask_path):
        arr = np.array(Image.open(mask_path).convert("RGB"))
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        if np.any((g > 100) & (r < 80) & (b < 80)):
            return "benign"
        if np.any((r > 100) & (g < 80) & (b < 80)):
            return "malignant"
        return "normal"

    images_dir = folder / "images"
    masks_dir = folder / "masks"
    entries = []
    for name in sorted(os.listdir(images_dir)):
        entries.append({
            "image": f"./images/{name}",
            "pathology": label_from_mask(masks_dir / name),
            "clinical_data": "",
        })
    _write_json(folder / "dataset.json", entries)


def organize_blui(folder):
    entries = []
    for cls, subdir in [("benign", "Benign"), ("malignant", "Malignant")]:
        for name in sorted(os.listdir(folder / subdir)):
            if name.endswith("Image.bmp"):
                entries.append({
                    "image": f"./{subdir}/{name}",
                    "pathology": cls,
                    "clinical_data": "",
                })
    _write_json(folder / "dataset.json", entries)


def organize_busi(folder):
    entries = []
    for cls in ("benign", "malignant", "normal"):
        for name in sorted(os.listdir(folder / cls)):
            # original images are named "<class> (N).png"; masks have "_mask" suffix
            if name.endswith(").png"):
                entries.append({
                    "image": f"./{cls}/{name}",
                    "pathology": cls,
                    "clinical_data": "",
                })
    _write_json(folder / "dataset.json", entries)


def organize_buscot(folder):
    source_json = folder / "extracted/BUSCoT/DatasetFiles/lesion_dataset.json"
    if not source_json.exists():
        print(f"[BUS-CoT] {source_json} not found; skipping organize step.")
        return

    with open(source_json) as f:
        raw = json.load(f)

    elasto_map = {
        "2分": "2 score",
        "4分": "4 score",
        "偏硬": "Slightly hard",
        "硬": "Hard",
        "质硬": "Hard tissue",
    }
    descriptor_keys = [
        "LesionEdge", "LesionBoundary", "LesionCalcificationFeatures",
        "EchoCharacteristics", "BloodFlowFeatures", "ElastographyFeatures",
    ]

    out = []
    for _, info in raw.items():
        pathology = info.get("pathology_histology", {}).get("pathology", "").lower()
        if pathology not in ("benign", "malignant"):
            continue
        img_rel = info.get("image_path", "")
        if not img_rel:
            continue

        us = info.get("us_report", {})
        parts = []
        for k in descriptor_keys:
            if k == "BloodFlowFeatures":
                v = us.get("BloodFlowFeaturesStr", us.get(k, ""))
            elif k == "ElastographyFeatures":
                v = us.get("ElastographyFeaturesStr", us.get(k, ""))
                v = elasto_map.get(v, v) if v else ""
            else:
                v = us.get(k, "")
            if v:
                parts.append(f"{k}: {v}")

        out.append({
            "image": f"extracted/BUSCoT/{img_rel}",
            "pathology": pathology,
            "clinical_data": ", ".join(parts),
        })

    _write_json(folder / "dataset.json", out)


def _write_json(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Each entry maps a dataset name to:
#   folder         : subdirectory under datasets/
#   urls           : {downloaded_filename: (url, expected_size_or_None)}
#   extract        : {zip_filename: {src_prefix_in_zip: dst_prefix_in_folder}}
#   organize       : function that builds dataset.json from the extracted tree
DATASETS = {
    "BrEaST": {
        "folder": "2024 - BrEaST",
        "urls": {
            "BrEaST_ds.zip": (
                "https://www.cancerimagingarchive.net/wp-content/uploads/"
                "BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip", None),
            "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx": (
                "https://www.cancerimagingarchive.net/wp-content/uploads/"
                "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx", None),
        },
        "extract": {"BrEaST_ds.zip": {"BrEaST-Lesions_USG-images_and_masks": "images_and_masks"}},
        "organize": organize_breast,
    },
    "BUSBRA": {
        "folder": "2024 - BUSBRA",
        "urls": {"BUSBRA_ds.zip": ("https://zenodo.org/records/8231412/files/BUSBRA.zip?download=1", None)},
        "extract": {"BUSBRA_ds.zip": {"BUSBRA/Images": "Images", "BUSBRA/bus_data.csv": "bus_data.csv"}},
        "organize": organize_busbra,
    },
    "BUS-UCLM": {
        "folder": "2024 - BUS-UCLM",
        "urls": {"BUSUCLM_ds.zip": (
            "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/7fvgj4jsp7-1.zip", None)},
        "extract": {"BUSUCLM_ds.zip": {
            "BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM/images": "images",
            "BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM/masks": "masks",
        }},
        "organize": organize_busuclm,
    },
    "BLUI": {
        "folder": "2023 - BLUI",
        "urls": {
            "BLUI_Benign_ds.zip":    ("https://qamebi.com/wp-content/uploads/2022/11/Benign.zip", None),
            "BLUI_Malignant_ds.zip": ("https://qamebi.com/wp-content/uploads/2022/11/Malignant.zip", None),
        },
        "extract": {
            "BLUI_Benign_ds.zip":    {"Benign": "Benign"},
            "BLUI_Malignant_ds.zip": {"Malignant": "Malignant"},
        },
        "organize": organize_blui,
    },
    "BUSI": {
        "folder": "2020 - BUSI",
        "urls": {"BUSI_ds.zip": ("https://scholar.cu.edu.eg/Dataset_BUSI.zip", None)},
        "extract": {"BUSI_ds.zip": {
            "Dataset_BUSI_with_GT/benign":    "benign",
            "Dataset_BUSI_with_GT/malignant": "malignant",
            "Dataset_BUSI_with_GT/normal":    "normal",
        }},
        "organize": organize_busi,
    },
    "BUS-CoT": {
        "folder": "2026 - BUS-CoT",
        # Figshare deposit accompanying Yu et al., 2026 (Sci. Data).
        "urls": {"BUSCoT.zip": ("https://ndownloader.figshare.com/files/60240116", 6942371401)},
        "extract": {"BUSCoT.zip": {"BUSCoT": "extracted/BUSCoT"}},
        "organize": organize_buscot,
    },
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_dataset(name, spec, datasets_root, skip_download, skip_extract):
    folder = datasets_root / spec["folder"]
    folder.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        for fname, (url, size) in spec["urls"].items():
            target = folder / fname
            if target.suffix == ".zip" and is_valid_zip(target) and \
                    (size is None or target.stat().st_size == size):
                continue
            download_file(url, target, expected_size=size)

    if not skip_extract:
        for zip_name, mapping in spec["extract"].items():
            zip_path = folder / zip_name
            if not zip_path.exists():
                tqdm.write(f"[{name}] zip missing: {zip_path}")
                continue
            extract_zip(zip_path, folder, mapping)

    spec["organize"](folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default=str(DEFAULT_DATASETS_DIR),
                        help=f"Root datasets directory (default: {DEFAULT_DATASETS_DIR}).")
    parser.add_argument("--only", nargs="+", choices=list(DATASETS.keys()),
                        help="Only process the named datasets.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Do not contact the network; assume zips are present.")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip the extract step (e.g., if already extracted).")
    args = parser.parse_args()

    datasets_root = Path(args.datasets).resolve()
    selected = args.only or list(DATASETS.keys())

    with tqdm(selected, desc="Datasets", unit="dataset") as bar:
        for name in bar:
            bar.set_postfix_str(name)
            process_dataset(name, DATASETS[name], datasets_root,
                            args.skip_download, args.skip_extract)

    print("All requested datasets processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
