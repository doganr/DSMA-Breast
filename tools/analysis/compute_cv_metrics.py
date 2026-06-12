import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dsma"))

import numpy as np
import torch
from utils.metrics import MetricsHandler
from utils.reporting import save_statistical_summary


def get_val(history, path):
    try:
        v = history
        for p in path: v = v[p]
        return v
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Recompute 5-fold cross-validation metrics "
                                                 "from a completed training run.")
    parser.add_argument("model_dir", help="Path to a saved training run (contains model_data.pth).")
    parser.add_argument("--output_dim", type=int, default=2,
                        help="Number of output classes used during training (default: 2).")
    args = parser.parse_args()

    model_dir = args.model_dir
    data_pth = os.path.join(model_dir, "model_data.pth")
    if not os.path.exists(data_pth):
        print(f"File not found: {data_pth}")
        return

    data = torch.load(data_pth, map_location='cpu', weights_only=False)

    best_fold_predictions = data['folds_predictions']
    global_history = data['global_history']
    
    # Optional: overwrite the loss to use the average loss as was done in train.py
    # This was handled in train.py so we just use the existing global_history which already has it usually
    # Or we can recalculate loss later if needed

    fold_results = {}
    fold_count = len(best_fold_predictions)

    for fold_key, preds_dict in best_fold_predictions.items():
        metrices = MetricsHandler(args.output_dim)
        metrices.y_preds = preds_dict['y_preds']
        metrices.y_preds_proba = preds_dict['y_preds_proba']
        metrices.y_trues = preds_dict['y_trues']
        fold_results[fold_key] = metrices.compute()

    calculated_metrics = {}
    keys_to_track = [
        ('ACC', ['accuracy']),
        ('AUROC', ['auc', 'macro']),
        ('AUPRC', ['aupr', 'macro']),
        ('F1', ['f1', 'macro']),
        ('Sens.', ['recall', 'macro']),
        ('Prec.', ['precision', 'macro'])
    ]

    for m_name, path in keys_to_track:
        extracted = [get_val(fold_results[f'fold{i}'], path) for i in range(1, fold_count + 1)]
        extracted = [x for x in extracted if x is not None]
        if extracted:
            calculated_metrics[m_name] = {
                'mean': float(np.mean(extracted)),
                'std': float(np.std(extracted)),
                'values': extracted
            }

    print("\n==========================================================")
    print("5-Fold Cross-Validation Aggregated Metrics")
    print("--- Mean ± Std (Across 5 Folds) ---")
    for k, v in calculated_metrics.items():
        print(f"  {k:10}: {v['mean']:.4f} ± {v['std']:.4f}")

    print("\n--- Global / Concatenated Truths (Overall Pool) ---")
    print(f"  Accuracy: {global_history['accuracy']:.4f}")
    if 'auc' in global_history:
        print(f"  AUC-ROC:  {global_history['auc']['macro']:.4f}")
    print(f"  F1-Score: {global_history['f1']['macro']:.4f}")
    
    # Repopulate the root metrics_summary files with these robust std values
    save_statistical_summary(global_history, model_dir, calculated_metrics)
    print(f"\n[Reporting] Repopulated metrics_summary.json/csv with real K-Fold Mean±Std in:")
    print(f"    {model_dir}")
    print("==========================================================\n")

if __name__ == '__main__':
    main()
