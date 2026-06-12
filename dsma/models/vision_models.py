"""
This module implements vision-related components for a multi-modal deep learning pipeline,
focusing on Vision Transformer (ViT) models and image preprocessing.
"""

import torch
from torchvision import transforms
from torch import optim, nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup

from utils.loss_functions import FocalLoss, FocalTverskyLoss

def initialize_transforms_old(args):
    input_size = 224
    img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image_transforms = {}
    image_transforms['train'] = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    image_transforms['test'] = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    return image_transforms

def initialize_transforms(args):
    input_size = 224
    img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image_transforms = {}

    image_transforms['train'] = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),  
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomRotation(degrees=10), 
        transforms.ColorJitter(brightness=0.1, contrast=0.2),  
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.5)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])

    image_transforms['test'] = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    
    return image_transforms

def get_vision_model(args, evaluation=False):
    device = torch.device('cuda', args.gpu)
    vision_transforms = initialize_transforms(args)
    vision_model = VIT_MODEL(args.vision_model, args.output_dim)  
    vision_model.to(device)

    if evaluation:
        vision_args ={'model':vision_model,
                    'transform':vision_transforms}
    else:
        vision_optimizer, vision_scheduler, vision_criterion = get_vision_configuration(args, vision_model)
        vision_args ={'model':vision_model,
                    'transform':vision_transforms, 
                    'optimizer':vision_optimizer,
                    'scheduler':vision_scheduler,
                    'criterion':vision_criterion}
    return vision_args

def get_vision_configuration(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.vision_lr, weight_decay=args.vision_weight_decay)
    num_training_steps = int(args.train_set_len / args.batch_size * args.epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    criterion = FocalLoss() #nn.CrossEntropyLoss()
    return optimizer, scheduler, criterion    

class VIT_MODEL(nn.Module):
    def __init__(self, vision_model, output_dim):
        super().__init__()
        self.vit = AutoModel.from_pretrained(vision_model)
        self.config = AutoConfig.from_pretrained(vision_model)
        self.classifier = nn.Linear(self.config.hidden_size, output_dim)
        
    def forward(self, image):
        outputs = self.vit(image, return_dict=True)
        
        # Swin Transformer outputs (B, L, C) without a direct pooler_output sometimes, 
        # or it requires different handling than ViT.
        if hasattr(self.config, 'model_type') and self.config.model_type == 'swin':
            # Swin's last_hidden_state is (B, L, C). We don't skip the first token (cls token) 
            # because Swin does not use a CLS token by default!
            embeddings = outputs.last_hidden_state
            # For classification, usually average pooling over the sequence (patch) dimension
            pooler_output = embeddings.mean(dim=1)
        else:
            # Standard ViT/DeiT behavior (skip CLS token for embeddings)
            embeddings = outputs.last_hidden_state[:, 1:, :]
            pooler_output = outputs.pooler_output
            
        res = {'embeddings': embeddings, 
               'cls': self.classifier(pooler_output)}
        return res

