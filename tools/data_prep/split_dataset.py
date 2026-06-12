import json
import os
import shutil

base_dir = 'datasets/2026 - BUS-CoT'
old_path = os.path.join(base_dir, 'dataset.json')
backup_path = os.path.join(base_dir, 'old_dataset.json')

if not os.path.exists(backup_path):
    shutil.copy2(old_path, backup_path)
    print(f'Backup created: {backup_path}')
else:
    print(f'Backup already exists: {backup_path}')

with open(backup_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

trainval_data = []
test_data = []

for item in data:
    split_val = item.get('split', 'unknown')
    if split_val == 'unknown':
        image_path = item.get('image', '')
        if 'trainval' in image_path:
            split_val = 'trainval'
        elif 'test' in image_path:
            split_val = 'test'
            
    if split_val == 'trainval':
        trainval_data.append(item)
    elif split_val == 'test':
        test_data.append(item)

with open(old_path, 'w', encoding='utf-8') as f:
    json.dump(trainval_data, f, indent=4)
print(f'dataset.json written (trainval): {len(trainval_data)} images')

with open(os.path.join(base_dir, 'dataset_test.json'), 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4)
print(f'dataset_test.json written (test): {len(test_data)} images')
print(f'Total images read: {len(data)}')
