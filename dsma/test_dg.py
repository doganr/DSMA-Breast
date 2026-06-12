import os
import argparse
import numpy as np
import torch
import copy
import sys
from tqdm import tqdm

from models.vision_models import get_vision_model
from models.text_models import get_text_model
from models.multimodal_models import get_multimodal_model
from data.ultrasound_dataset import MultiModalDataLoader
from utils.metrics import MetricsHandler

def test_on_domain(args, domain, device):
    print(f"\n--- Testing on domain: {domain} ---")
    args.include_dirs = [domain]
    
    # fold=0 to return a single test_loader for the entire specified domain
    try:
        test_loader = MultiModalDataLoader(args, fold=0, verbose=False)
    except Exception as e:
        print(f"Error loading {domain}: {e}")
        return None
        
    fold_metrics = []
    
    for fold in range(1, 6):
        model_path = os.path.join(args.model_dir, f'model_fold{fold}_best.pth')
        if not os.path.exists(model_path):
            print(f"  Warning: {model_path} not found.")
            continue
            
        print(f"  Loading {model_path} ...", end='')
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        args.evaluation = True
        
        models = {
            'multimodal': get_multimodal_model(args, evaluation=True),
            'vision': get_vision_model(args, evaluation=True),
            'text': get_text_model(args, evaluation=True)
        }
        
        models['vision']['model'].load_state_dict(checkpoint['vision'])
        models['vision']['model'].to(device).eval()
        
        models['text']['model'].load_state_dict(checkpoint['text'])
        models['text']['model'].to(device).eval()
        
        models['multimodal']['model'].load_state_dict(checkpoint['multimodal'])
        models['multimodal']['model'].to(device).eval()
        print(" Done.")
        
        metrics = MetricsHandler(args.output_dim)
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"    Eval Fold {fold} on {domain}", leave=False):
                vision_data, text_data, label = data[0].to(device), data[1], data[2].to(device)
                
                # Evaluation Mode -> single output
                preds = models['multimodal']['model'](models, vision_data, text_data, label, device)
                
                metrics.update(label, preds)
                
        history = metrics.compute()
        
        acc = history.get('accuracy', 0)
        pre = history.get('precision', {}).get('macro', 0)
        auc = history.get('auc', {}).get('macro', 0)
        rec = history.get('recall', {}).get('macro', 0)
        f1  = history.get('f1', {}).get('macro', 0)
        
        fold_metrics.append({
            'ACC': acc, 'PRE': pre, 'AUC': auc, 'REC': rec, 'F1': f1
        })
        
    if not fold_metrics:
        return None
        
    # Compute mean and std
    results = {}
    for key in ['ACC', 'PRE', 'AUC', 'REC', 'F1']:
        vals = [f[key] for f in fold_metrics]
        results[key] = f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--vision_model', default='facebook/deit-base-patch16-224')
    parser.add_argument('--text_model', default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--dataset', default='./datasets')
    parser.add_argument('--json_file', default='dataset.json')
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--class_names', nargs='+', default=['benign', 'malignant'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--multimodal_fusion', default='product')
    parser.add_argument('--multilevel_fusion', default='concat')
    parser.add_argument('--fusion_stages', default='both')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda', args.gpu)
    
    domains = ['2020 - BUSI', '2023 - BLUI', '2024 - BUS-UCLM', '2024 - BUSBRA']
    domain_results = {}
    
    for domain in domains:
        args_copy = copy.deepcopy(args)
        res = test_on_domain(args_copy, domain, device)
        if res:
            domain_results[domain] = res
            
    print("\n" + "="*85)
    print(f"{'Dataset':<20} | {'ACC (mean±std)':<15} | {'PRE (mean±std)':<15} | {'AUC (mean±std)':<15} | {'REC (mean±std)':<15} | {'F1 (mean±std)':<15}")
    print("-" * 85)
    for domain, res in domain_results.items():
        print(f"{domain:<20} | {res['ACC']:<15} | {res['PRE']:<15} | {res['AUC']:<15} | {res['REC']:<15} | {res['F1']:<15}")
    print("="*85)
