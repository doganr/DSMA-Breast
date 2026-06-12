import os
import time
import json
import warnings
import logging
import argparse
import torch
from tqdm import tqdm

from models.vision_models import get_vision_model
from models.text_models import get_text_model
from models.multimodal_models import get_multimodal_model
from data.ultrasound_dataset import MultiModalDataLoader
from utils.metrics import MetricsHandler

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model_dir', type=str, default='./dsma/saved/20251225_093410/model_fold2_best.pth')
    parser.add_argument('--vision_model', type=str, default='google/vit-base-patch16-224') 
    parser.add_argument('--text_model', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--evaluation', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='../datasets')
    parser.add_argument('--exclude_dirs', type=str, nargs='*', 
                        default=['2020 - BUSI','2022 - CVA-BUS','2023 - BLUI','2024 - BUSBRA','2024 - BUS-UCLM'],
                        help='Directories to exclude')
    # '2020 - BUSI','2023 - BLUI','2024 - BrEaST','2024 - BUS-UCLM', '2024 - BUSBRA'
    parser.add_argument('--output_dim', type=int, default=3)
    parser.add_argument('--class_names', type=str, nargs='*', 
                        default=['benign', 'malignant', 'normal'],
                        help='Class names')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--multimodal_fusion', type=str, default='product')
    parser.add_argument('--multilevel_fusion', type=str, default='concat')  
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

    args = parser.parse_args()

    device = torch.device('cuda', args.gpu)    

    print('Checkpoint loading...')

    checkpoint = torch.load(args.model_dir, weights_only=True)

    print('Checkpoint loaded...')

    models = {'multimodal':get_multimodal_model(args, evaluation=True),
              'vision':get_vision_model(args, evaluation=True),
              'text':get_text_model(args, evaluation=True)}

    print('Vision model loading...')
    models['vision']['model'].load_state_dict(checkpoint['vision'])  
    models['vision']['model'].to(device)
    models['vision']['model'].eval()  
    print('Vision model loaded...')

    print('Text model loading...')
    models['text']['model'].load_state_dict(checkpoint['text'])  
    models['text']['model'].to(device)
    models['text']['model'].eval()  
    print('Text model loaded...')

    print('Multimodal model loading...')
    models['multimodal']['model'].load_state_dict(checkpoint['multimodal'])  
    models['multimodal']['model'].to(device)
    models['multimodal']['model'].eval()  
    print('Multimodal model loaded...')

    test_loader = MultiModalDataLoader(args, fold=0, verbose=False)

    evaluation_metrics = MetricsHandler(args.output_dim)
    with tqdm(test_loader, unit="batch", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=False) as tepoch:              
        tepoch.set_description(f"Evaluating")
        for iter, data in enumerate(tepoch):
            vision_data, text_data, label = data[0], data[1], data[2]     

            predictions = models['multimodal']['model'](models, vision_data, text_data, label, device)

            evaluation_metrics.update(label, predictions)
    
    evaluation_history = evaluation_metrics.compute()

    print('Accuracy:', evaluation_history['accuracy'])
    print('Precision:', evaluation_history['precision']['macro'])
    print('Recall:', evaluation_history['recall']['macro'])
    print('F1:', evaluation_history['f1']['macro'])
    print('AUC:', evaluation_history['auc']['macro'])
    print('AUPR:', evaluation_history['aupr']['macro'])

    