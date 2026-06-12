"""
Post-Training Report Generator
===============================
Generates publication-ready reports (CSV, plots, Grad-CAM) from already-completed
training runs. Reads `model_data.pth` and `global_metrics.json` from the experiment folder.

Usage:
    python3 generate_reports.py <experiment_folder_path> [--class_names benign malignant normal]
    
Example:
    python3 generate_reports.py ./dsma/saved/multimodal_deit-base-patch16-224_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_BrEaST_BUS-CoT_0303_2215
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

# Add the dsma/ package to path so we can import utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dsma"))

from utils.reporting import save_statistical_summary, plot_confusion_matrix, plot_roc_auc, plot_learning_curves

def main():
    parser = argparse.ArgumentParser(description="Generate reports from completed training runs")
    parser.add_argument('experiment_dir', type=str, help='Path to the experiment folder (contains model_data.pth)')
    parser.add_argument('--class_names', type=str, nargs='*', default=['benign', 'malignant', 'normal'])
    args = parser.parse_args()
    
    exp_dir = args.experiment_dir
    model_data_path = os.path.join(exp_dir, 'model_data.pth')
    global_metrics_path = os.path.join(exp_dir, 'global_metrics.json')
    
    if not os.path.exists(model_data_path):
        print(f"ERROR: model_data.pth not found in {exp_dir}")
        return
    
    print(f"Loading training data from: {exp_dir}")
    
    # Load saved training data
    model_data = torch.load(model_data_path, map_location='cpu', weights_only=False)
    all_train_history = model_data['folds_history']
    best_fold_predictions = model_data['folds_predictions']
    
    # Load global metrics
    with open(global_metrics_path, 'r') as f:
        global_history = json.load(f)
    
    # Reconstruct global y_trues, y_preds, y_preds_proba from fold predictions
    y_trues = []
    y_preds = []
    y_preds_proba = []
    
    for fold_key, fold_data in best_fold_predictions.items():
        y_trues += fold_data['y_trues']
        y_preds += fold_data['y_preds']
        if 'y_preds_proba' in fold_data:
            y_preds_proba += fold_data['y_preds_proba']
    
    print(f"  Total samples: {len(y_trues)}")
    print(f"  Folds: {len(all_train_history)}")
    print(f"  Epochs per fold: {len(all_train_history[0])}")
    print(f"  Proba available: {len(y_preds_proba) > 0}")
    
    # Auto-detect number of classes from probability shape
    if y_preds_proba:
        num_classes = len(y_preds_proba[0])
    else:
        num_classes = len(set(y_trues))
    
    # Auto-generate class names if not matching
    ALL_CLASS_MAP = {2: ['benign', 'malignant'], 3: ['benign', 'malignant', 'normal']}
    if len(args.class_names) != num_classes:
        args.class_names = ALL_CLASS_MAP.get(num_classes, [f'class_{i}' for i in range(num_classes)])
    
    print(f"  Detected classes: {num_classes} -> {args.class_names}")
    
    # Compute approximate final validation loss for the summary
    global_history['loss'] = np.mean([f[-1]['evaluation']['loss'] for f in all_train_history])
    
    # A) Statistical Summary (global)
    print("\n[1/5] Generating metrics_summary.csv & .json ...")
    save_statistical_summary(global_history, exp_dir)
    
    # A2) Per-Fold Mean ± Std Summary (publication-ready)
    print("[2/5] Generating per-fold Mean ± Std summary ...")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import label_binarize
    import pandas as pd
    
    num_classes = len(args.class_names)
    fold_metrics = []
    
    for fold_key in sorted(best_fold_predictions.keys()):
        fd = best_fold_predictions[fold_key]
        yt = np.array(fd['y_trues'])
        yp = np.array(fd['y_preds'])
        
        acc = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, average='macro', zero_division=0)
        rec = recall_score(yt, yp, average='macro', zero_division=0)
        f1 = f1_score(yt, yp, average='macro', zero_division=0)
        
        auc_val = 0.0
        if 'y_preds_proba' in fd and len(fd['y_preds_proba']) > 0:
            try:
                ypp = np.array(fd['y_preds_proba'])
                if num_classes == 2:
                    # Binary: use probability of positive class directly
                    auc_val = roc_auc_score(yt, ypp[:, 1])
                else:
                    # Multiclass: use OVR
                    yt_bin = label_binarize(yt, classes=range(num_classes))
                    auc_val = roc_auc_score(yt_bin, ypp, average='macro', multi_class='ovr')
            except Exception:
                auc_val = 0.0
        
        val_loss = all_train_history[int(fold_key.replace('fold','')) - 1][-1]['evaluation']['loss']
        
        fold_metrics.append({
            'Fold': fold_key, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1': f1, 'AUC': auc_val, 'Val_Loss': val_loss
        })
    
    df_folds = pd.DataFrame(fold_metrics)
    
    # Compute Mean ± Std row
    mean_row = {col: f"{df_folds[col].mean():.4f} ± {df_folds[col].std():.4f}" 
                for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Val_Loss']}
    mean_row['Fold'] = 'Mean ± Std'
    
    # Format fold values to 4 decimal
    for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Val_Loss']:
        df_folds[col] = df_folds[col].apply(lambda x: f"{x:.4f}")
    
    df_folds = pd.concat([df_folds, pd.DataFrame([mean_row])], ignore_index=True)
    
    csv_path = os.path.join(exp_dir, 'metrics_summary_folds.csv')
    df_folds.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(f"\n  {df_folds.to_string(index=False)}\n")
    
    # B) Confusion Matrix
    print("[3/5] Generating confusion matrix ...")
    plot_confusion_matrix(y_trues, y_preds, args.class_names, exp_dir)
    
    # C) ROC AUC Curve
    if y_preds_proba:
        print("[4/5] Generating ROC AUC curve ...")
        plot_roc_auc(y_trues, y_preds_proba, args.class_names, exp_dir)
    else:
        print("[4/5] SKIPPED: No probability data found for ROC AUC.")
    
    # D) Learning Curves
    print("[5/5] Generating learning curves ...")
    plot_learning_curves(all_train_history, exp_dir)
    
    print(f"\nAll reports saved to: {exp_dir}")
    print("Generated files:")
    for f in sorted(os.listdir(exp_dir)):
        if f.endswith(('.csv', '.json', '.png')):
            print(f"  - {f}")

if __name__ == "__main__":
    main()
