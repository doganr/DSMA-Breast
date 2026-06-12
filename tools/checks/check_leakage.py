"""
Perceptual-hash (pHash) duplicate scan between BUS-CoT and the other five
datasets, used to verify that the metadata-based leakage filter is
comprehensive. The script reports any image pair with a Hamming distance
below the configured threshold and writes the full list to
leakage_report.json next to this file.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

try:
    import imagehash
except ImportError:
    print("Installing imagehash...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imagehash"])
    import imagehash


def get_images_from_dataset(dataset_dir, include_dirs=None, exclude_dirs=None):
    """Collect (path, domain, label) tuples from every dataset.json under dataset_dir."""
    dataset_files = list(Path(dataset_dir).rglob("dataset.json"))
    image_paths = []

    for file in dataset_files:
        domain = file.parent.name

        if include_dirs and domain not in include_dirs:
            continue
        if exclude_dirs and domain in exclude_dirs:
            continue

        with open(file, 'r') as f:
            data = json.load(f)

        base_dir = str(file.parent)
        for row in data:
            img_subpath = row['image']
            if img_subpath.startswith('./'):
                img_subpath = img_subpath[2:]

            full_path = os.path.join(base_dir, img_subpath).replace('\\', '/')
            if os.path.exists(full_path):
                image_paths.append((full_path, domain, row['pathology']))

    return image_paths


def main():
    print("Starting cross-dataset leakage scan...")
    dataset_path = "./datasets"

    print("Loading BUS-CoT images...")
    buscot_data = get_images_from_dataset(dataset_path, include_dirs=["2026 - BUS-CoT"])
    print(f"BUS-CoT image count: {len(buscot_data)}")

    print("Loading the other datasets (excluding BUS-CoT)...")
    other_data = get_images_from_dataset(dataset_path, exclude_dirs=["2026 - BUS-CoT"])
    print(f"Other-datasets image count: {len(other_data)}")

    if len(buscot_data) == 0 or len(other_data) == 0:
        print("No images found. Please verify the dataset directory names.")
        return

    print("\nComputing pHash for BUS-CoT...")
    buscot_hashes = {}
    for path, domain, label in tqdm(buscot_data):
        try:
            img = Image.open(path).convert('RGB')
            file_hash = imagehash.phash(img)
            buscot_hashes[path] = {'hash': file_hash, 'domain': domain, 'label': label}
        except Exception:
            pass

    print("\nComputing pHash for the other datasets...")
    other_hashes = {}
    for path, domain, label in tqdm(other_data):
        try:
            img = Image.open(path).convert('RGB')
            file_hash = imagehash.phash(img)
            other_hashes[path] = {'hash': file_hash, 'domain': domain, 'label': label}
        except Exception:
            pass

    # Hamming distance interpretation:
    #   0     identical image
    #   1-3   minor differences (crop, resampling, watermark, etc.)
    #   > 5   different image
    threshold = 2

    print(f"\nComparing image pairs (Hamming distance <= {threshold})...")
    leaked_pairs = []

    for b_path, b_info in tqdm(buscot_hashes.items()):
        b_hash = b_info['hash']
        for o_path, o_info in other_hashes.items():
            distance = b_hash - o_info['hash']
            if distance <= threshold:
                leaked_pairs.append({
                    'buscot_img': b_path,
                    'other_img': o_path,
                    'other_domain': o_info['domain'],
                    'distance': distance,
                    'buscot_label': b_info['label'],
                    'other_label': o_info['label'],
                })

    print("\n" + "=" * 50)
    print("LEAKAGE REPORT")
    print("=" * 50)
    print(f"Near-duplicate pairs detected: {len(leaked_pairs)}")

    if len(leaked_pairs) > 0:
        print("\nFirst 10 examples:")
        for pair in leaked_pairs[:10]:
            print(f"- [distance={pair['distance']}] BUS-CoT ({pair['buscot_label']}) "
                  f"<=> {pair['other_domain']} ({pair['other_label']})")
            print(f"    BUS-CoT: {pair['buscot_img']}")
            print(f"    Other  : {pair['other_img']}\n")

        # Write the report next to this script so the output stays
        # co-located with the source, independent of the invoking cwd.
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "leakage_report.json")
        with open(report_path, "w") as f:
            json.dump(leaked_pairs, f, indent=4)
        print(f"Full report saved to {report_path}")
    else:
        print(f"\nNo cross-dataset duplicates found between BUS-CoT and the other datasets "
              f"(threshold = {threshold}).")


if __name__ == "__main__":
    main()
