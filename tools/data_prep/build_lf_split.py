"""
Build leakage-free (LF) BUS-CoT splits.

Removes BUS-CoT entries whose BUS-Expert annotation marks the source
dataset as BUSBRA or BUSI (those datasets are evaluated independently
in the pooled experiments). Two modes: trainval and test.
"""

import argparse
import json
import os


def _format_features(us_report):
    keys = [
        "LesionEdge",
        "LesionBoundary",
        "LesionCalcificationFeatures",
        "EchoCharacteristics",
        "BloodFlowFeatures",
        "ElastographyFeatures",
    ]
    return ", ".join(f"{k}: {us_report[k]}" for k in keys if k in us_report)


def build_trainval(base_dir, excluded):
    lesion_file = os.path.join(base_dir, "extracted/BUSCoT/DatasetFiles/lesion_dataset.json")
    expert_file = os.path.join(base_dir, "extracted/BUSCoT/DatasetFiles/BUS-Expert_dataset.json")

    print("Loading original datasets...")
    with open(lesion_file, "r", encoding="utf-8") as f:
        buscot = json.load(f)
    with open(expert_file, "r", encoding="utf-8") as f:
        busexpert = json.load(f)

    lf_list = []
    removed = {name: 0 for name in excluded}
    kept = 0

    print("Processing and filtering...")
    for key, entry in buscot.items():
        pk = key.split("_")[1].split("@")[0]
        source_dataset = busexpert.get(pk, {}).get("dataset", "Unknown")

        if source_dataset in excluded:
            removed[source_dataset] += 1
            continue

        kept += 1
        lf_list.append({
            "image": f"extracted/BUSCoT/{entry['image_path']}",
            "pathology": entry["pathology_histology"]["pathology"].lower(),
            "clinical_data": _format_features(entry.get("us_report", {})),
            "source_dataset": source_dataset,
        })

    out_file = os.path.join(base_dir, "dataset_lf.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(lf_list, f, indent=4)

    print("====================================")
    print("Leakage-Free Trainval Split Complete")
    print(f"Total Original Images : {len(buscot)}")
    for name, n in removed.items():
        print(f"Removed {name:<7}: {n}")
    print(f"Remaining Safe Images : {kept}")
    print(f"Saved to              : {out_file}")
    print("====================================")


def build_test(base_dir, excluded):
    test_file = os.path.join(base_dir, "dataset_test.json")
    expert_file = os.path.join(base_dir, "extracted/BUSCoT/DatasetFiles/BUS-Expert_dataset.json")

    print("Loading test set...")
    with open(test_file, "r", encoding="utf-8") as f:
        buscot_test = json.load(f)
    with open(expert_file, "r", encoding="utf-8") as f:
        busexpert = json.load(f)

    lf_test_list = []
    removed = {name: 0 for name in excluded}
    kept = 0

    print("Filtering test set...")
    for entry in buscot_test:
        pk = entry["image"].split("/")[-1].split("@")[0]
        source_dataset = busexpert.get(pk, {}).get("dataset", "Unknown")

        if source_dataset in excluded:
            removed[source_dataset] += 1
            continue

        kept += 1
        lf_test_list.append(entry)

    out_file = os.path.join(base_dir, "dataset_test_lf.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(lf_test_list, f, indent=4)

    print("====================================")
    print("Leakage-Free Test Split Complete")
    print(f"Total Original Test Images : {len(buscot_test)}")
    for name, n in removed.items():
        print(f"Removed {name:<7}: {n}")
    print(f"Remaining Safe Test Images : {kept}")
    print(f"Saved to                   : {out_file}")
    print("====================================")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["trainval", "test", "both"], default="both",
                        help="Which LF split to build (default: both).")
    parser.add_argument("--base_dir", default="datasets/2026 - BUS-CoT",
                        help="Root of the BUS-CoT dataset.")
    parser.add_argument("--exclude", nargs="+", default=["BUSBRA", "BUSI"],
                        help="Source-dataset labels (from BUS-Expert) to drop.")
    args = parser.parse_args()

    excluded = set(args.exclude)
    if args.mode in ("trainval", "both"):
        build_trainval(args.base_dir, excluded)
    if args.mode in ("test", "both"):
        build_test(args.base_dir, excluded)


if __name__ == "__main__":
    main()
