"""
Inspect the fold structure of a completed training run alongside the
StratifiedKFold partition that would be produced on the current data.
Useful for verifying that saved predictions are aligned with the
expected fold split.
"""

import argparse
import collections
import os
import sys

import torch
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dsma"))
from data.ultrasound_dataset import ReadDatasets

DEFAULT_INCLUDE_DIRS = [
    "2020 - BUSI", "2023 - BLUI", "2024 - BUS-UCLM",
    "2024 - BUSBRA", "2024 - BrEaST", "2026 - BUS-CoT",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Path to a saved training run (contains model_data.pth).")
    parser.add_argument("--dataset", default="datasets",
                        help="Root datasets directory (default: datasets).")
    parser.add_argument("--json_file", default="dataset.json",
                        help="Per-dataset JSON file to read (default: dataset.json).")
    parser.add_argument("--include_dirs", nargs="+", default=DEFAULT_INCLUDE_DIRS,
                        help="Dataset directories included during training.")
    parser.add_argument("--class_names", nargs="+", default=["benign", "malignant"],
                        help="Class names matching the training run.")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=61, help="StratifiedKFold random_state.")
    args = parser.parse_args()

    data_pth = os.path.join(args.model_dir, "model_data.pth")
    if not os.path.exists(data_pth):
        print(f"File not found: {data_pth}")
        return

    d = torch.load(data_pth, weights_only=False)
    preds = d["folds_predictions"]

    print("Saved fold sizes:")
    for i in range(1, args.folds + 1):
        key = f"fold{i}"
        if key in preds:
            print(f"  {key} y_trues: {len(preds[key]['y_trues'])}")

    ds_args = argparse.Namespace(
        class_names=args.class_names,
        dataset=args.dataset,
        json_file=args.json_file,
        include_dirs=args.include_dirs,
    )
    data, labels = ReadDatasets(ds_args, include_dirs=args.include_dirs)
    print(f"\nCurrent dataset length: {len(data)}")
    print("Domains distribution:", collections.Counter([x["domain"] for x in data]))

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    print(f"\nRecomputed StratifiedKFold (seed={args.seed}):")
    for i, (_, test_idx) in enumerate(skf.split(range(len(data)), labels)):
        print(f"  Fold {i+1} test indices: {len(test_idx)}")


if __name__ == "__main__":
    main()
