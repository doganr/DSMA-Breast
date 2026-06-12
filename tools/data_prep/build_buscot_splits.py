import json
import os

base_dir = "datasets/2026 - BUS-CoT"
source_file = os.path.join(base_dir, "extracted/BUSCoT/DatasetFiles/lesion_dataset.json")

with open(source_file, "r") as f:
    data = json.load(f)

trainval_list = []
test_list = []

for key, entry in data.items():
    # Build clinical data string
    us_r = entry.get("us_report", {})
    features = []
    if "LesionEdge" in us_r: features.append(f"LesionEdge: {us_r['LesionEdge']}")
    if "LesionBoundary" in us_r: features.append(f"LesionBoundary: {us_r['LesionBoundary']}")
    if "LesionCalcificationFeatures" in us_r: features.append(f"LesionCalcificationFeatures: {us_r['LesionCalcificationFeatures']}")
    if "EchoCharacteristics" in us_r: features.append(f"EchoCharacteristics: {us_r['EchoCharacteristics']}")
    if "BloodFlowFeatures" in us_r: features.append(f"BloodFlowFeatures: {us_r['BloodFlowFeatures']}")
    if "ElastographyFeatures" in us_r: features.append(f"ElastographyFeatures: {us_r['ElastographyFeatures']}")
    
    clinical_data_str = ", ".join(features)
    
    formatted_entry = {
        "image": f"extracted/BUSCoT/{entry['image_path']}",
        "pathology": entry['pathology_histology']['pathology'].lower(),
        "clinical_data": clinical_data_str
    }
    
    if entry.get("split") == "trainval":
        trainval_list.append(formatted_entry)
    elif entry.get("split") == "test":
        test_list.append(formatted_entry)

# Restore old_dataset.json and write the new ones
os.rename(os.path.join(base_dir, "dataset.json"), os.path.join(base_dir, "old_dataset.json"))

with open(os.path.join(base_dir, "dataset.json"), "w") as f:
    json.dump(trainval_list, f, indent=4)

with open(os.path.join(base_dir, "dataset_test.json"), "w") as f:
    json.dump(test_list, f, indent=4)

print(f"Total images: {len(data)}")
print(f"Trainval (dataset.json): {len(trainval_list)}")
print(f"Test (dataset_test.json): {len(test_list)}")
