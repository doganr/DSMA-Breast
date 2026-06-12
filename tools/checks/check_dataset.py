"""
Quick dataset-integrity checks.

Subcommands:
  sources : BUS-CoT image source attribution audit (per BUS-Expert).
  sizes   : per-domain sample count via the project data loader.
  train   : raw benign/malignant count in each dataset_train.json.
"""

import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DSMA_DIR = os.path.join(REPO_ROOT, "dsma")

DOMAINS = [
    "2020 - BUSI",
    "2023 - BLUI",
    "2024 - BUS-UCLM",
    "2024 - BUSBRA",
    "2024 - BrEaST",
    "2026 - BUS-CoT",
]


def cmd_sources(args):
    """Source attribution audit for BUS-CoT entries."""
    print("Loading datasets...")
    with open(os.path.join(args.datasets, "2026 - BUS-CoT/dataset.json")) as f:
        dataset_json = json.load(f)
    print(f"dataset.json total images: {len(dataset_json)}")

    expert_path = os.path.join(
        args.datasets,
        "2026 - BUS-CoT/extracted/BUSCoT/DatasetFiles/BUS-Expert_dataset.json",
    )
    with open(expert_path) as f:
        expert_json = json.load(f)
    print("BUS-Expert_dataset.json loaded.")

    source_counts = {}
    missing = 0

    for row in dataset_json:
        match = re.search(r"(\d+)@", row["image"])
        if match and match.group(1) in expert_json:
            source = expert_json[match.group(1)].get("dataset", "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        else:
            missing += 1

    print("\n=================================")
    print("BUS-CoT source attribution:")
    print("=================================")
    for k, v in sorted(source_counts.items(), key=lambda item: -item[1]):
        print(f"- {k}: {v} images")
    if missing:
        print(f"\nUnmatched / unknown ID: {missing} images")


def cmd_sizes(args):
    """Verify each domain loads through the project data loader."""
    sys.path.insert(0, DSMA_DIR)
    from data.ultrasound_dataset import ReadDatasets  # noqa: E402
    from collections import Counter

    fake_args = argparse.Namespace(
        class_names=["benign", "malignant"],
        dataset=args.datasets,
        json_file=args.json_file,
    )
    data, _ = ReadDatasets(fake_args, include_dirs=DOMAINS)

    print(f"Total samples in {args.json_file}: {len(data)}")
    print("Domain distribution:", Counter(d["domain"] for d in data))


def cmd_train(args):
    """Raw JSON count of benign/malignant per dataset_train.json."""
    total = 0
    for d in DOMAINS:
        fp = os.path.join(args.datasets, d, args.json_file)
        if not os.path.exists(fp):
            print(f"{fp} does not exist.")
            continue
        with open(fp) as f:
            entries = json.load(f)
        n = sum(1 for x in entries if x["pathology"] in ("benign", "malignant"))
        print(f"{d} {args.json_file}: {n}")
        total += n
    print(f"Total: {total}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="datasets",
                        help="Root datasets directory (default: ./datasets).")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    subparsers.add_parser("sources", help="BUS-CoT source attribution audit")

    p_sizes = subparsers.add_parser("sizes", help="Loadable sample count per domain")
    p_sizes.add_argument("--json_file", default="dataset_train.json")

    p_train = subparsers.add_parser("train", help="Raw benign/malignant count per dataset_train.json")
    p_train.add_argument("--json_file", default="dataset_train.json")

    args = parser.parse_args()
    {"sources": cmd_sources, "sizes": cmd_sizes, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
