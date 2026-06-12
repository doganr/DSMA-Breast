"""
This module provides a `MetricsHandler` class to compute various evaluation metrics 
for multi-class classification tasks in PyTorch. It integrates metrics from 
scikit-learn and supports performance evaluation for both batch-level and 
overall dataset-level predictions.
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import torch

class MetricsHandler:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.y_trues = []
        self.y_preds = []
        self.y_preds_proba = []

    def roc_aupr_score(self, y_true, y_score, average="macro"):
        def _binary_roc_aupr_score(y_true, y_score):
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            return auc(recall, precision)

        def _average_binary_score(binary_metric, y_true, y_score, average):
            if average == "binary":
                return binary_metric(y_true, y_score)
            if average == "micro":
                y_true = y_true.ravel()
                y_score = y_score.ravel()
            if y_true.ndim == 1:
                y_true = y_true.reshape((-1, 1))
            if y_score.ndim == 1:
                y_score = y_score.reshape((-1, 1))
            n_classes = y_score.shape[1]
            score = np.zeros((n_classes,))
            for c in range(n_classes):
                y_true_c = y_true.take([c], axis=1).ravel()
                y_score_c = y_score.take([c], axis=1).ravel()
                score[c] = binary_metric(y_true_c, y_score_c)
            return np.average(score)
        return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

    def update(self, y_true, y_pred):
        y_true_idx = torch.max(y_true, 1)[1]
        y_pred_idx = torch.max(y_pred.data, 1)[1]
        probs = torch.softmax(y_pred.data, dim=1)

        self.y_trues += list(y_true_idx.detach().cpu().numpy())
        self.y_preds += list(y_pred_idx.detach().cpu().numpy())
        self.y_preds_proba += list(probs.detach().cpu().numpy())

    def compute(self, is_batch=False):
        y_trues_one_hot = label_binarize(
            self.y_trues, classes=np.arange(self.num_classes))
        y_preds_one_hot = label_binarize(
            self.y_preds, classes=np.arange(self.num_classes))
        metrices = {
            "accuracy": accuracy_score(self.y_trues, self.y_preds),
            "precision": {'micro': precision_score(self.y_trues, self.y_preds, average='micro', zero_division=0),
                          'macro': precision_score(self.y_trues, self.y_preds, average='macro', zero_division=0)},
            "recall": {'micro': recall_score(self.y_trues, self.y_preds, average='micro', zero_division=0),
                       'macro': recall_score(self.y_trues, self.y_preds, average='macro', zero_division=0)},
            "f1": {'micro': f1_score(self.y_trues, self.y_preds, average='micro', zero_division=0),
                   'macro': f1_score(self.y_trues, self.y_preds, average='macro', zero_division=0)}
        }
        if not is_batch:
            y_proba = np.array(self.y_preds_proba)
            if self.num_classes == 2:
                # Binary: use probability of positive class (class 1)
                y_trues_arr = np.array(self.y_trues)
                metrices["auc"] = {'micro': roc_auc_score(y_trues_arr, y_proba[:, 1]),
                                   'macro': roc_auc_score(y_trues_arr, y_proba[:, 1])}
                metrices["aupr"] = {'micro': self.roc_aupr_score(y_trues_one_hot, y_proba[:, 1], average='micro'),
                                   'macro': self.roc_aupr_score(y_trues_one_hot, y_proba[:, 1], average='macro')}
            else:
                # Multiclass: use full probability matrix
                metrices["auc"] = {'micro': roc_auc_score(y_trues_one_hot, y_proba, average='micro'),
                                   'macro': roc_auc_score(y_trues_one_hot, y_proba, average='macro')}
                metrices["aupr"] = {'micro': self.roc_aupr_score(y_trues_one_hot, y_proba, average='micro'),
                                   'macro': self.roc_aupr_score(y_trues_one_hot, y_proba, average='macro')}

        return metrices

    def reset(self):
        self.y_trues = []
        self.y_preds = []
        self.y_preds_proba = []