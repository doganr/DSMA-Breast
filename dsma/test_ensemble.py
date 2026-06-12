import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

from models.vision_models import initialize_transforms, get_vision_model
from models.text_models import initialize_tokenizer, get_text_model
from models.multimodal_models import get_multimodal_model
from data.ultrasound_dataset import MultiModalDataLoader
from utils.metrics import MetricsHandler
from utils.reporting import plot_confusion_matrix, plot_roc_auc, save_statistical_summary

if __name__ == "__main__":
    device = torch.device('cuda', 0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', type=str, default='multimodal')
    parser.add_argument('--vision_model', type=str, default='google/vit-base-patch16-224')
    parser.add_argument('--text_model', type=str, default='google-bert/bert-base-uncased')
    parser.add_argument('--fusion_stages', type=str, default='both')
    parser.add_argument('--model_dir', type=str, required=True)
    p_args = parser.parse_args()

    args = argparse.Namespace(
        train_type=p_args.train_type,
        vision_model=p_args.vision_model,
        text_model=p_args.text_model,
        dataset='../datasets',
        include_dirs=['2026 - BUS-CoT'],
        output_dim=2,
        class_names=['benign', 'malignant'],
        batch_size=20,
        num_workers=8,
        fmap=768,
        multimodal_fusion='product',
        multilevel_fusion='concat',
        fusion_stages=p_args.fusion_stages,
        json_file='dataset_test.json',
        gpu=0,
        evaluation=False, # to simulate the train.py environment properly
        multimodal_lr=1e-5,
        multimodal_weight_decay=5e-6,
        vision_lr=1e-5,
        vision_weight_decay=5e-6,
        text_lr=1e-5,
        text_weight_decay=5e-6,
        train_set_len=4338,
        epoch=20
    )

    model_dir = p_args.model_dir
    test_dir = os.path.join(model_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    print("Loading test dataset from dataset_test.json...")
    test_loader = MultiModalDataLoader(args, fold=0, verbose=False)
    
    ensemble_y_preds_proba = None
    test_y_trues = None
    
    fold_count = 5
    fold_results = {}
    
    for fold in range(1, fold_count + 1):
        print(f"\nLoading fold {fold} model...")
        checkpoint_path = os.path.join(model_dir, f"model_fold{fold}_best.pth")
        
        if args.train_type == 'vision':
            models = {'vision': get_vision_model(args)}
        elif args.train_type == 'text':
            models = {'text': get_text_model(args)}
        else:
            models = {
                'multimodal': get_multimodal_model(args),
                'vision': get_vision_model(args),
                'text': get_text_model(args)
            }
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        for key in models:
            if key in checkpoint:
                models[key]['model'].load_state_dict(checkpoint[key])
            models[key]['model'].to(device)
            models[key]['model'].eval()

        metrices = MetricsHandler(args.output_dim)
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"Evaluating Fold {fold}"):
                vision_data, text_data, label = data[0], data[1], data[2]
                
                if args.train_type == 'vision':
                    vision_outputs = models['vision']['model'](vision_data.to(device))
                    preds = vision_outputs['cls']
                elif args.train_type == 'text':
                    text_data_input = models['text']['tokenizer'](text_data, padding='longest', truncation=True, return_tensors='pt')['input_ids'].to(device)
                    text_outputs = models['text']['model'](text_data_input)
                    preds = text_outputs['cls']
                else:
                    vision_data = vision_data.to(device)
                    preds, mm_v_loss, mm_t_loss, mm_c_loss = models['multimodal']['model'](models, vision_data, text_data, label, device)
                    
                metrices.update(label, preds)

        fold_history = metrices.compute()
        fold_results[f'fold_{fold}'] = fold_history

        if ensemble_y_preds_proba is None:
            ensemble_y_preds_proba = np.array(metrices.y_preds_proba)
            test_y_trues = metrices.y_trues
        else:
            ensemble_y_preds_proba += np.array(metrices.y_preds_proba)
            
    # Calculate ensemble probabilities
    ensemble_y_preds_proba /= fold_count
    ensemble_y_preds = np.argmax(ensemble_y_preds_proba, axis=1)

    # Compute final metrics
    final_metrices = MetricsHandler(args.output_dim)
    final_metrices.y_preds = ensemble_y_preds.tolist()
    final_metrices.y_preds_proba = ensemble_y_preds_proba.tolist()
    final_metrices.y_trues = test_y_trues
    
    test_history = final_metrices.compute()
    
    # Process fold-wise statistics
    calculated_metrics = {}
    keys_to_track = [
        ('ACC', ['accuracy']),
        ('AUROC', ['auc', 'macro']),
        ('AUPRC', ['aupr', 'macro']),
        ('F1', ['f1', 'macro']),
        ('Sens.', ['recall', 'macro']),
        ('Prec.', ['precision', 'macro'])
    ]
    
    # Safely extract
    def get_val(history, path):
        try:
            v = history
            for p in path: v = v[p]
            return v
        except:
            return None

    for m_name, path in keys_to_track:
        extracted = [get_val(fold_results[f'fold_{i}'], path) for i in range(1, fold_count + 1)]
        extracted = [x for x in extracted if x is not None]
        if extracted:
            calculated_metrics[m_name] = {
                'mean': float(np.mean(extracted)),
                'std': float(np.std(extracted)),
                'values': extracted
            }
    
    combined_results = {
        'fold_results': fold_results,
        'mean_std_metrics': calculated_metrics,
        'ensemble_results': test_history
    }
    
    # Save results
    with open(os.path.join(test_dir, "test_metrics_folds.json"), 'w') as f:
        json.dump(combined_results, f, indent=4)
        
    save_statistical_summary(test_history, test_dir, calculated_metrics)
    plot_confusion_matrix(final_metrices.y_trues, final_metrices.y_preds, args.class_names, test_dir)
    plot_roc_auc(final_metrices.y_trues, final_metrices.y_preds_proba, args.class_names, test_dir)
    
    print(f"\n==========================================================")
    print(f"Independent test evaluation completed.")
    
    print("\n--- Mean ± Std (Across 5 Folds) ---")
    for k, v in calculated_metrics.items():
        print(f"  {k:10}: {v['mean']:.4f} ± {v['std']:.4f}")
        
    print(f"\n--- Ensemble Voting Result ---")    
    print(f"  Accuracy: {test_history['accuracy']:.4f}")
    if 'auc' in test_history:
        print(f"  AUC-ROC:  {test_history['auc']['macro']:.4f}")
    print(f"  F1-Score: {test_history['f1']['macro']:.4f}")
    print(f"[Reporting] Independent test results saved in:\n    {test_dir}")
    print(f"==========================================================\n")
