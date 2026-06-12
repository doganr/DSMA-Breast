"""
This module implements a multi-modal data loading pipeline for ultrasound image datasets,
supporting stratified k-fold cross-validation and dataset preparation for training and testing.
"""

import os
import json
from pathlib import Path
import random
from collections import Counter
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

from models.vision_models import initialize_transforms
from models.text_models import initialize_tokenizer

def MultiModalDataLoader(args, fold=5, verbose=True):
    image_transforms = initialize_transforms(args)
    text_tokenizer = initialize_tokenizer(args)
    dict_data, labels = ReadDatasets(args, include_dirs=args.include_dirs)

    # number_of_images    

    if fold>0:
        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=args.seed)
        loaders = []

        for fold, (train_indices, test_indices) in enumerate(skf.split(range(len(dict_data)), labels)):
            args.train_set_len = ((len(train_indices) // args.batch_size) * args.epoch)/.8

            train_data = [dict_data[i] for i in train_indices]
            test_data = [dict_data[i] for i in test_indices]

            if verbose:        
                train_labels = [labels[i] for i in train_indices]
                test_labels = [labels[i] for i in test_indices]
                label_counts = Counter(labels)
                train_label_counts = Counter(train_labels)
                test_label_counts = Counter(test_labels)
                print(f'Fold {fold+1}:')
                print(f'Total images: {label_counts}')
                print(f'Train images: {train_label_counts}')
                print(f'Test images: {test_label_counts}')

            train_set = UltrasoundDataSet(args, train_data, transform=image_transforms['train'])
            test_set = UltrasoundDataSet(args, test_data, transform=image_transforms['test'])

            # Handle domain imbalance via WeightedRandomSampler
            train_domains = [d['domain'] for d in train_data]
            domain_counts = Counter(train_domains)
            sample_weights = [1.0 / domain_counts[d] for d in train_domains]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

            train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            loaders.append([train_loader, test_loader])
        
        return loaders
    else:
        test_set = UltrasoundDataSet(args, dict_data, transform=image_transforms['test'])
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        return test_loader
    
def ReadDatasets(args, include_dirs=[]):
    def read_json_file(file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)  
        return data
    datasets_path = args.dataset
    target_json = getattr(args, 'json_file', 'dataset.json')

    # Define base dataset locations by searching for dataset.json
    base_dataset_files = list(Path(datasets_path).rglob("dataset.json"))
    
    if include_dirs:
        base_dataset_files = [path for path in base_dataset_files if any(inc in path.parts for inc in include_dirs)]    
    
    dataset_files = []
    for base_file in base_dataset_files:
        specific_file = base_file.parent / target_json
        if specific_file.exists():
            used_json_name = target_json
            final_path = specific_file
        else:
            used_json_name = 'dataset.json'
            final_path = base_file
        dataset_files.append(('./'+str(final_path).replace("\\","/"), used_json_name))
    
    all_data = []
    for file, used_json_name in dataset_files:
        try:
            for row in read_json_file(file):
                if row['pathology'] in args.class_names:
                    # Clean the base dir path (e.g. './datasets/2024 - BrEaST/')
                    base_dir = file.replace(used_json_name, '')
                    # Image path from JSON (e.g. './Images/case.png' or 'extracted/BUSCoT/img.png')
                    img_subpath = row['image']
                    if img_subpath.startswith('./'):
                        img_subpath = img_subpath[2:]
                    
                    full_image_path = os.path.join(base_dir, img_subpath).replace('\\', '/')
                    
                    domain = Path(file).parent.name
                    all_data.append({
                        'image': full_image_path,
                        'pathology': row['pathology'],
                        'clinical_data': row['clinical_data'] if len(row['clinical_data']) > 0 else 'No data.',
                        'domain': domain
                    })
        except Exception as e:
            raw_data = read_json_file(file)
            print(f"Error reading {file}: {e}. Data type: {type(raw_data)}")
            raise e
    labels = [data['pathology'] for data in all_data]
    return all_data, labels

class UltrasoundDataSet(Dataset):
    def __init__(self, args, data, transform=None): 
        self.args = args
        self.transform = transform        
        self.data = data
    
    def __len__(self):
        return len(self.data)   

    def __getitem__(self, idx):
        class_labels = {label:np.eye(len(self.args.class_names))[i] for i, label in enumerate(self.args.class_names)}
        #class_labels = {'benign':np.eye(3)[0], 'malignant':np.eye(3)[1], 'normal':np.eye(3)[2]}
        if isinstance(idx, slice):
            return [(self._load_image(f['image']), f['clinical_data'], class_labels[f['pathology']]) for f in self.data[idx]]
        else:
            return self._load_image(self.data[idx]['image']), self.data[idx]['clinical_data'], class_labels[self.data[idx]['pathology']]
    
    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

