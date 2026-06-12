import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch
import torch.nn.functional as F

def save_statistical_summary(global_history, save_dir, mean_std_metrics=None):
    """Saves the final Mean metrics to CSV and JSON."""
    def _extract(val, key='macro'):
        """Extract float from nested dict or return float directly."""
        if isinstance(val, dict):
            return val.get(key, 0.0)
        return val
    
    acc = _extract(global_history.get('accuracy', 0.0))
    prec = _extract(global_history.get('precision', 0.0))
    rec = _extract(global_history.get('recall', 0.0))
    f1 = _extract(global_history.get('f1', 0.0))
    loss = _extract(global_history.get('loss', 0.0))
    auc_val = _extract(global_history.get('auc', 0.0))
    aupr_val = _extract(global_history.get('aupr', 0.0))
    
    metrics_names = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)', 'AUC (macro)', 'AUPR (macro)', 'Loss']
    values = [
        f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{auc_val:.4f}", f"{aupr_val:.4f}", f"{loss:.4f}"
    ]

    summary_dict = {
        'Metric': metrics_names,
        'Ensemble_Value': values
    }

    if mean_std_metrics:
        # Expected keys from mean_std_metrics: ACC, Prec., Sens., F1, AUROC, AUPRC
        mean_std_mapping = {
            'Accuracy': 'ACC',
            'Precision (macro)': 'Prec.',
            'Recall (macro)': 'Sens.',
            'F1-Score (macro)': 'F1',
            'AUC (macro)': 'AUROC',
            'AUPR (macro)': 'AUPRC'
        }
        mean_std_col = []
        
        # Determine how many folds exist by checking one of the valid keys
        num_folds = 0
        for ms_key in mean_std_mapping.values():
            if ms_key in mean_std_metrics:
                num_folds = len(mean_std_metrics[ms_key].get('values', []))
                break
                
        # Initialize fold columns dynamically
        fold_columns = {f"Fold_{i+1}": [] for i in range(num_folds)}
        
        for mn in metrics_names:
            ms_key = mean_std_mapping.get(mn)
            if ms_key and ms_key in mean_std_metrics:
                m_val = mean_std_metrics[ms_key]['mean']
                s_val = mean_std_metrics[ms_key]['std']
                vals = mean_std_metrics[ms_key].get('values', [])
                
                mean_std_col.append(f"{m_val:.4f} ± {s_val:.4f}")
                
                for i in range(num_folds):
                    if i < len(vals):
                        fold_columns[f"Fold_{i+1}"].append(f"{float(vals[i]):.4f}")
                    else:
                        fold_columns[f"Fold_{i+1}"].append("N/A")
            else:
                mean_std_col.append("N/A")
                for i in range(num_folds):
                    fold_columns[f"Fold_{i+1}"].append("N/A")
                    
        summary_dict['KFold_Mean_Std'] = mean_std_col
        for fold_name, fold_data in fold_columns.items():
            summary_dict[fold_name] = fold_data

    df = pd.DataFrame(summary_dict)
    df.to_csv(os.path.join(save_dir, 'metrics_summary.csv'), index=False)
    
    with open(os.path.join(save_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(summary_dict, f, indent=4)
    
    print(f"[Reporting] Saved metrics_summary.csv & .json")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir, filename="global_confusion_matrix.png"):
    """Plots and saves a Seaborn confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()


def plot_learning_curves(all_train_history, save_dir):
    """Plots Train/Val Loss and Accuracy with std deviation bands across folds."""
    num_folds = len(all_train_history)
    if num_folds == 0: return
    num_epochs = len(all_train_history[0])
    
    train_loss = np.zeros((num_folds, num_epochs))
    val_loss = np.zeros((num_folds, num_epochs))
    train_acc = np.zeros((num_folds, num_epochs))
    val_acc = np.zeros((num_folds, num_epochs))
    
    for f in range(num_folds):
        for e in range(num_epochs):
            train_loss[f, e] = all_train_history[f][e]['train']['loss']
            val_loss[f, e] = all_train_history[f][e]['evaluation']['loss']
            train_acc[f, e] = all_train_history[f][e]['train']['accuracy']
            val_acc[f, e] = all_train_history[f][e]['evaluation']['accuracy']
            
    epochs = np.arange(1, num_epochs + 1)
    
    # Create CSV records
    records = []
    for f in range(num_folds):
        for e in range(num_epochs):
            records.append({
                'Fold': f + 1, 'Epoch': e + 1,
                'Train_Loss': train_loss[f, e], 'Val_Loss': val_loss[f, e],
                'Train_Acc': train_acc[f, e], 'Val_Acc': val_acc[f, e]
            })
    pd.DataFrame(records).to_csv(os.path.join(save_dir, 'learning_curves.csv'), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.set_style("whitegrid")
    
    def plot_metric(ax, data, title, ylabel):
        mean_data = np.mean(data, axis=0)
        std_data = np.std(data, axis=0)
        
        for f in range(num_folds):
            ax.plot(epochs, data[f], linestyle='--', alpha=0.5, label=f'Fold {f}')
            
        ax.plot(epochs, mean_data, color='black', linewidth=2, label='Mean Score')
        ax.fill_between(epochs, mean_data - std_data, mean_data + std_data, alpha=0.2, label='Std Dev (±1)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabel)
        ax.legend()
        
        # Annotate final mean value
        ax.text(epochs[-1], mean_data[-1], f"{mean_data[-1]:.4f}", fontweight='bold')

    plot_metric(axes[0, 0], train_loss, 'Training Loss', 'Loss')
    plot_metric(axes[0, 1], train_acc, 'Training Accuracy', 'Accuracy')
    plot_metric(axes[1, 0], val_loss, 'Validation Loss', 'Loss')
    plot_metric(axes[1, 1], val_acc, 'Validation Accuracy', 'Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'), dpi=300)
    plt.close()


def plot_roc_auc(y_trues, y_preds_proba, class_names, save_dir):
    """Plots ROC AUC curve for multiclass predictions."""
    try:
        from sklearn.preprocessing import label_binarize
        
        y_trues_bin = label_binarize(y_trues, classes=range(len(class_names)))
        y_preds_proba = np.array(y_preds_proba)
        
        plt.figure(figsize=(10, 8))
        sns.set_style("whitegrid")
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, class_name in enumerate(class_names):
            if i < y_trues_bin.shape[1] and i < y_preds_proba.shape[1]:
                fpr, tpr, _ = roc_curve(y_trues_bin[:, i], y_preds_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                         label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - {len(class_names)}class')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_auc_curve.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Failed to plot ROC AUC: {e}")
