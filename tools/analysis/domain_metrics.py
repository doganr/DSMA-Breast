"""
Per-Domain Metrics Calculator
==============================
Computes per-dataset (domain) metrics from already-completed training runs.
Uses the EXACT same ReadDatasets function and StratifiedKFold split to
reconstruct fold assignments, then maps predictions back to source domains.

Usage:
    python3 domain_metrics.py <experiment_folder> --dataset ./datasets \
        --include_dirs "2024 - BrEaST" "2026 - BUS-CoT" \
        --class_names benign malignant \
        --json_file dataset_lf.json \
        --seed 61
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Use the EXACT same ReadDatasets from training
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dsma"))
from data.ultrasound_dataset import ReadDatasets


def main():
    parser = argparse.ArgumentParser(description="Compute per-domain metrics from completed training")
    parser.add_argument('experiment_dir', type=str)
    parser.add_argument('--dataset', type=str, default='./datasets')
    parser.add_argument('--include_dirs', type=str, nargs='*', 
                        default=['2024 - BrEaST', '2026 - BUS-CoT'])
    parser.add_argument('--class_names', type=str, nargs='*', 
                        default=['benign', 'malignant'])
    parser.add_argument('--json_file', type=str, default='dataset_lf.json',
                        help='Which JSON to read. BUS-CoT uses dataset_lf.json, others fallback to dataset.json')
    parser.add_argument('--seed', type=int, default=61)
    parser.add_argument('--fold', type=int, default=5)
    args = parser.parse_args()
    
    exp_dir = args.experiment_dir
    model_data_path = os.path.join(exp_dir, 'model_data.pth')
    
    if not os.path.exists(model_data_path):
        print(f"ERROR: model_data.pth not found in {exp_dir}")
        return
    
    # 1. Reload datasets using the EXACT same ReadDatasets from training
    print(f"Loading dataset metadata using ReadDatasets (json_file={args.json_file})...")
    dict_data, labels = ReadDatasets(args, include_dirs=args.include_dirs)
    print(f"  Total samples: {len(dict_data)}")
    print(f"  Domains: {Counter([d['domain'] for d in dict_data])}")
    
    # 2. Load saved predictions
    print(f"Loading predictions from: {exp_dir}")
    model_data = torch.load(model_data_path, map_location='cpu', weights_only=False)
    best_fold_predictions = model_data['folds_predictions']
    
    # Verify total matches
    total_preds = sum(len(best_fold_predictions[k]['y_trues']) for k in best_fold_predictions)
    print(f"  Total predictions across folds: {total_preds}")
    if total_preds != len(dict_data):
        print(f"  WARNING: Mismatch! Dataset has {len(dict_data)} samples but predictions have {total_preds}")
        print(f"  This may cause incorrect domain mapping!")
    else:
        print(f"  Match: {total_preds} predictions == {len(dict_data)} dataset samples")
    
    # 3. Reconstruct the EXACT same folds using same seed
    skf = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
    
    # Collect per-domain predictions across all folds
    domain_results = defaultdict(lambda: {'y_trues': [], 'y_preds': [], 'y_preds_proba': []})
    
    for fold_idx, (train_indices, test_indices) in enumerate(skf.split(range(len(dict_data)), labels)):
        fold_key = f'fold{fold_idx + 1}'
        if fold_key not in best_fold_predictions:
            print(f"  WARNING: {fold_key} not found in predictions, skipping")
            continue
            
        fold_preds = best_fold_predictions[fold_key]
        y_preds = fold_preds['y_preds']
        y_trues = fold_preds['y_trues']
        y_preds_proba = fold_preds.get('y_preds_proba', [])
        
        print(f"  {fold_key}: test_indices={len(test_indices)}, y_trues={len(y_trues)}, y_preds={len(y_preds)}")
        
        # Map each test sample to its domain
        for i, test_idx in enumerate(test_indices):
            if i >= len(y_trues):
                break
            domain = dict_data[test_idx]['domain']
            domain_results[domain]['y_trues'].append(y_trues[i])
            domain_results[domain]['y_preds'].append(y_preds[i])
            if y_preds_proba:
                domain_results[domain]['y_preds_proba'].append(y_preds_proba[i])
    
    # 4. Compute per-domain metrics
    print(f"\n{'='*90}")
    print(f"{'Domain':<25} {'N':>6} {'Accuracy':>10} {'Prec(macro)':>12} {'Rec(macro)':>12} {'F1(macro)':>10} {'AUC(macro)':>12}")
    print(f"{'='*90}")
    
    results_for_csv = []
    num_classes = len(args.class_names)
    
    for domain in sorted(domain_results.keys()):
        data = domain_results[domain]
        yt = np.array(data['y_trues'])
        yp = np.array(data['y_preds'])
        
        n = len(yt)
        acc = accuracy_score(yt, yp)
        
        prec = precision_score(yt, yp, average='macro', zero_division=0, labels=range(num_classes))
        rec = recall_score(yt, yp, average='macro', zero_division=0, labels=range(num_classes))
        f1 = f1_score(yt, yp, average='macro', zero_division=0, labels=range(num_classes))
        
        # AUC needs probability data and multiple classes present
        auc_str = "N/A"
        auc_val = 0.0
        if data['y_preds_proba']:
            try:
                ypp = np.array(data['y_preds_proba'])
                unique_classes = sorted(set(yt))
                if len(unique_classes) >= 2:
                    if num_classes == 2:
                        auc_val = roc_auc_score(yt, ypp[:, 1])
                    else:
                        yt_bin = label_binarize(yt, classes=range(num_classes))
                        auc_val = roc_auc_score(yt_bin, ypp, average='macro', multi_class='ovr', labels=unique_classes)
                    auc_str = f"{auc_val:.4f}"
                else:
                    auc_str = "1-cls"
            except Exception as e:
                print(f"    Failed AUC calculation for {domain}: {e}")
                auc_str = "N/A"
        
        print(f"{domain:<25} {n:>6} {acc:>10.4f} {prec:>12.4f} {rec:>12.4f} {f1:>10.4f} {auc_str:>12}")
        
        results_for_csv.append({
            'Domain': domain,
            'N': n,
            'Accuracy': f"{acc:.4f}",
            'Precision_macro': f"{prec:.4f}",
            'Recall_macro': f"{rec:.4f}",
            'F1_macro': f"{f1:.4f}",
            'AUC_macro': auc_str
        })
    
    print(f"{'='*90}")
    
    # 5. Save CSV
    csv_path = os.path.join(exp_dir, 'per_domain_metrics.csv')
    pd.DataFrame(results_for_csv).to_csv(csv_path, index=False)
    print(f"\nPer-domain metrics saved to: {csv_path}")
    
    # 6. Also save per-domain details as JSON
    detail_path = os.path.join(exp_dir, 'per_domain_metrics.json')
    json_results = {}
    for domain in sorted(domain_results.keys()):
        data = domain_results[domain]
        yt = data['y_trues']
        yp = data['y_preds']
        class_dist = {args.class_names[c]: int(np.sum(np.array(yt) == c)) for c in range(num_classes) if c in yt}
        json_results[domain] = {
            'total_samples': len(yt),
            'class_distribution': class_dist,
            'accuracy': float(accuracy_score(yt, yp)),
            'precision_macro': float(precision_score(yt, yp, average='macro', zero_division=0, labels=range(num_classes))),
            'recall_macro': float(recall_score(yt, yp, average='macro', zero_division=0, labels=range(num_classes))),
            'f1_macro': float(f1_score(yt, yp, average='macro', zero_division=0, labels=range(num_classes)))
        }
    
    with open(detail_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    print(f"Per-domain details saved to: {detail_path}")


if __name__ == "__main__":
    main()
