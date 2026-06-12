"""
This module implements two custom loss functions for use in deep learning models:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, target):
        if target.ndim > 1 and target.size(1) > 1:  
            target = torch.argmax(target, dim=1)

        target = target.to(torch.int64)

        log_probs = F.log_softmax(preds, dim=1) 
        probs = torch.exp(log_probs)             

        target_log_probs = log_probs[range(len(target)), target]  
        target_probs = probs[range(len(target)), target]          

        focal_loss = -self.alpha * ((1 - target_probs) ** self.gamma) * target_log_probs  

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, reduction='mean'):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, target):
        if target.ndim > 1 and target.size(1) > 1:  
            target = torch.argmax(target, dim=1)

        preds = F.softmax(preds, dim=1)
        
        target_one_hot = F.one_hot(target, num_classes=preds.size(1)).float() 

        TP = (preds * target_one_hot).sum(dim=(0, 1))  
        FP = ((1 - target_one_hot) * preds).sum(dim=(0, 1))  
        FN = (target_one_hot * (1 - preds)).sum(dim=(0, 1))  

        tversky_index = TP / (TP + self.alpha * FP + self.beta * FN + 1e-6)  

        focal_tversky_loss = (1 - tversky_index) ** self.gamma

        if self.reduction == 'mean':
            return focal_tversky_loss.mean()
        elif self.reduction == 'sum':
            return focal_tversky_loss.sum()
        else:
            return focal_tversky_loss