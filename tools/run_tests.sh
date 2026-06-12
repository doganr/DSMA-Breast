#!/bin/bash
# Example batch evaluation runner. Replace MODEL_DIR placeholders with
# the actual paths of your saved training runs. Works regardless of the
# invoking cwd.

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dsma-breast

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../dsma" || exit 1

# Edit the following list with the paths of your saved runs under ./saved/.
# Each entry is: "<train_type>|<vision_model>|<text_model>|<run_dir>"
RUNS=(
    "multimodal|google/vit-base-patch16-224|emilyalsentzer/Bio_ClinicalBERT|saved/<your_run_folder>"
    # Add more lines as needed.
)

for entry in "${RUNS[@]}"; do
    IFS='|' read -r TRAIN_TYPE VISION_MODEL TEXT_MODEL MODEL_DIR <<< "$entry"
    echo "Evaluating $(basename "$MODEL_DIR")"
    python3 test_ensemble.py \
        --train_type "$TRAIN_TYPE" \
        --vision_model "$VISION_MODEL" \
        --text_model "$TEXT_MODEL" \
        --model_dir "$MODEL_DIR"
done

echo "All evaluations finished."
