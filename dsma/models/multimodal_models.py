"""
This module implements a multi-modal framework combining vision and text data for classification tasks.
It includes model initialization, configuration, and a custom multi-modal neural network.
"""

import math
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import ViTForImageClassification, get_linear_schedule_with_warmup
from utils.loss_functions import FocalLoss, FocalTverskyLoss

def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):

    def _lr_lambda(current_step):

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def get_multimodal_model(args, evaluation=False):
    device = torch.device('cuda', args.gpu)
    multimodal_model = MultiModalBUC_Net(args)
    multimodal_model.to(device)
    if evaluation:
        mm_args = {'model':multimodal_model}
    else:
        mm_optimizer, mm_scheduler, mm_criterion = get_multimodal_configuration(args, multimodal_model)    
        mm_args = {'model':multimodal_model,
                'optimizer':mm_optimizer,
                'scheduler':mm_scheduler,
                'criterion':mm_criterion}
    return mm_args

def get_multimodal_configuration(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.multimodal_lr, weight_decay=args.multimodal_weight_decay)
    num_training_steps = int(args.train_set_len / args.batch_size * args.epoch)
    #scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_training_steps=num_training_steps)
    criterion = WeightedFocalLoss()
    return optimizer, scheduler, criterion
    
class WeightedFocalLoss(nn.Module):
    def __init__(self, vision_loss_weight=0.7, text_loss_weight=0.3, align_weight=0.1):
        super(WeightedFocalLoss, self).__init__()
        self.loss = FocalLoss()
        self.vision_loss_weight = vision_loss_weight
        self.text_loss_weight = text_loss_weight
        self.align_weight = align_weight

    def forward(self, predictions, label, mm_vision_loss, mm_text_loss, align_loss):
        classification_loss = self.loss(predictions, label)

        weighted_loss = classification_loss + \
                        (self.vision_loss_weight * mm_vision_loss) + \
                        (self.text_loss_weight * mm_text_loss) + \
                        (self.align_weight * align_loss)

        return weighted_loss

class MultiModalBUC_Net(nn.Module):
    def __init__(self, args):
        super().__init__()   
        self.args = args  

        self.evaluation = getattr(args, 'evaluation', False)
        self.multimodal_fusion = args.multimodal_fusion #'product'
        self.multilevel_fusion = args.multilevel_fusion #'concat'
        # 'lca_only' (Local Correlation Alignment), 'gaa_only' (Global Attention Alignment), 'both'
        # Legacy aliases 'ccm_only'/'mha_only' are accepted for backward compatibility with prior checkpoints.
        self.fusion_stages = getattr(args, 'fusion_stages', 'both')
        self.fmap = getattr(args, 'fmap', 768)

        self.ReLu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.classification_fc1 = nn.Linear(self.get_strategy_rate() * self.fmap, self.fmap, bias=True)
        self.classification_fc2 = nn.Linear(self.fmap, args.output_dim, bias=True)

        if self.fusion_stages in ('gaa_only', 'mha_only', 'both'):
            self.correlation_mhsa = nn.MultiheadAttention(embed_dim=self.fmap, num_heads=8, batch_first=True)

        if self.fusion_stages in ('lca_only', 'ccm_only', 'both'):
            self.correlation_conv = nn.Sequential(
                nn.Conv2d(1, 64, 3, stride=1, padding=1),
                nn.Conv2d(64, 1, 3, stride=1, padding=1),
                nn.ReLU()
            )

    def get_strategy_rate(self):
        strategies = {'concat':2, 'product':1, 'sum':1}
        assert (self.multilevel_fusion in strategies) and (self.multimodal_fusion in strategies), "Invalid fusion strategy"
        return strategies[self.multilevel_fusion] * strategies[self.multimodal_fusion]

    def fusion(self, embeddings1, embeddings2, strategy):        
        fusion_strategies = {
            'sum': lambda x, y: (x + y) / 2,
            'product': lambda x, y: x * y,
            'concat': lambda x, y: torch.cat([x, y], dim=1)
        }
        assert strategy in fusion_strategies, "Invalid fusion strategy"
        return fusion_strategies[strategy](embeddings1, embeddings2)


    def align_features(self, vision_embeddings, text_embeddings):
        aligned_vision_parts = []
        aligned_text_parts = []

        # Stage I: LCA
        if self.fusion_stages in ('lca_only', 'ccm_only', 'both'):
            attention_map = torch.bmm(vision_embeddings, text_embeddings.transpose(1, 2))
            attention_map = self.correlation_conv(attention_map.unsqueeze(1)).squeeze(1)
            # attention_map shape after squeeze: [B, N_v, N_t]
            # Row-wise / column-wise means produce per-token attention weights.
            vision_attention = attention_map.mean(dim=2)  # [B, N_v]
            text_attention = attention_map.mean(dim=1)    # [B, N_t]
            vision_attention, text_attention = torch.sigmoid(vision_attention), torch.sigmoid(text_attention)
            aligned_vision_parts.append(vision_attention.unsqueeze(-1) * vision_embeddings)
            aligned_text_parts.append(text_attention.unsqueeze(-1) * text_embeddings)

        # Stage II: GAA
        if self.fusion_stages in ('gaa_only', 'mha_only', 'both'):
            combined_embeddings = torch.cat((vision_embeddings, text_embeddings), dim=1)
            attn_output, _ = self.correlation_mhsa(combined_embeddings, combined_embeddings, combined_embeddings)
            aligned_vision_parts.append(attn_output[:, :vision_embeddings.size(1), :])
            aligned_text_parts.append(attn_output[:, vision_embeddings.size(1):, :])

        # Combine parts
        if len(aligned_vision_parts) == 1:
            aligned_vision_embeddings = aligned_vision_parts[0]
            aligned_text_embeddings = aligned_text_parts[0]
        else:
            aligned_vision_embeddings = torch.cat(aligned_vision_parts, dim=1)
            aligned_text_embeddings = torch.cat(aligned_text_parts, dim=1)

        return aligned_vision_embeddings, aligned_text_embeddings

    def calculate_semantic_embeddings(self, vision_pool, text_pool, batch_size):
        variance_vision = torch.nn.functional.normalize(torch.var(vision_pool, dim=0), dim=-1)
        variance_text = torch.nn.functional.normalize(torch.var(text_pool, dim=0), dim=-1)
        semantic_vision_embeddings = vision_pool + vision_pool * variance_vision.unsqueeze(0).repeat(batch_size, 1)
        semantic_text_embeddings = text_pool + text_pool * variance_text.unsqueeze(0).repeat(batch_size, 1)
        return semantic_vision_embeddings, semantic_text_embeddings

    def forward(self, models, vision_data, text_data, label, device):
        vision_data, label = vision_data.to(device), label.to(device)
        text_data = models['text']['tokenizer'](text_data, padding='longest', truncation=True, return_tensors='pt')['input_ids'].to(device)

        vision_outputs = models['vision']['model'](vision_data)
        text_outputs = models['text']['model'](text_data)
        vision_embeddings, text_embeddings = vision_outputs['embeddings'], text_outputs['embeddings']

        aligned_vision_embeddings, aligned_text_embeddings = self.align_features(vision_embeddings, text_embeddings)

        vision_pool = torch.mean(aligned_vision_embeddings, dim=1)
        text_pool = torch.mean(aligned_text_embeddings, dim=1)

        cosine_similarity = F.cosine_similarity(vision_pool, text_pool, dim=-1)
        align_loss = torch.mean(1 - cosine_similarity)

        # Semantic embedding
        semantic_vision_embeddings, semantic_text_embeddings = self.calculate_semantic_embeddings(vision_pool,
                                                                                                  text_pool,
                                                                                                  vision_embeddings.size(0))

        F_sem = self.fusion(semantic_vision_embeddings, semantic_text_embeddings, self.multimodal_fusion) #'product'
        F_pool = self.fusion(vision_pool, text_pool, self.multimodal_fusion) #'product'
        F_final = self.fusion(F_sem, F_pool, self.multilevel_fusion) #'concat'
        hidden_fc = self.dropout(self.ReLu(self.classification_fc1(F_final)))
        # Return logits; the loss (FocalLoss) and MetricsHandler each apply
        # softmax / log_softmax internally as appropriate.
        logits = self.classification_fc2(hidden_fc).squeeze()

        if self.evaluation:
            return logits
        else:
            text_classification_loss = models['text']['criterion'](text_outputs['cls'], label)
            vision_classification_loss = models['vision']['criterion'](vision_outputs['cls'], label)
            return logits, vision_classification_loss, text_classification_loss, align_loss