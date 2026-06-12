"""
This module provides components for initializing, configuring, and using text models,
specifically BERT-based models, for multi-modal or standalone text classification tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from utils.loss_functions import FocalLoss, FocalTverskyLoss

def initialize_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    return tokenizer

def get_text_model(args, embedding=True, evaluation=False):
    device = torch.device('cuda', args.gpu)
    text_tokenizer = initialize_tokenizer(args)
    text_model = BERT_MODEL(args.text_model, args.output_dim,  embedding=embedding) 
    text_model.to(device)
    if evaluation:
        text_args ={'model':text_model,
                    'tokenizer':text_tokenizer}
    else:
        text_optimizer, text_scheduler, text_criterion = get_text_configuration(args, text_model)
        text_args ={'model':text_model,
                    'tokenizer':text_tokenizer, 
                    'optimizer':text_optimizer,
                    'scheduler':text_scheduler,
                    'criterion':text_criterion}
    return text_args

def get_text_configuration(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.text_lr, weight_decay=args.text_weight_decay)
    num_training_steps = int(args.train_set_len / args.batch_size * args.epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    criterion = FocalLoss() #nn.CrossEntropyLoss()
    return optimizer, scheduler, criterion

class BERT_MODEL(nn.Module):
    def __init__(self, bert_model, output_dim, embedding=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.config = AutoConfig.from_pretrained(bert_model)
        self.classifier = nn.Linear(self.config.hidden_size, output_dim)
        self.embedding = embedding        
        
    def forward(self, text):
        outputs = self.bert(text)
        res = {'embeddings': outputs.last_hidden_state[:, 1:, :], 
               'cls': self.classifier(outputs.pooler_output)}
        return res