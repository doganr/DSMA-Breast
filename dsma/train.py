import os
import time
import json
import warnings
import logging
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from tqdm import tqdm

from models.vision_models import initialize_transforms, get_vision_model
from models.text_models import initialize_tokenizer, get_text_model
from models.multimodal_models import get_multimodal_model
from data.ultrasound_dataset import MultiModalDataLoader, ReadDatasets
from utils.metrics import MetricsHandler
from utils.reporting import plot_learning_curves, plot_roc_auc, plot_confusion_matrix, save_statistical_summary
from utils.gradcam import save_gradcam_samples

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(loader, models, train_metrices, device):
    epoch_loss = 0
    for key in models:
        models[key]['model'].train()   
    with tqdm(loader, unit="batch", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=False) as tepoch: 
        train_ftext = f"   └─ Train"              
        tepoch.set_description(f"{train_ftext:18}")
        for iter, data in enumerate(tepoch):
            vision_data, text_data, label = data[0], data[1], data[2]      

            if 'multimodal' in models:
                predictions, mm_vision_loss, mm_text_loss, align_loss = models['multimodal']['model'](models, vision_data, text_data, label, device)
                loss = models['multimodal']['criterion'](predictions, label.to(device), mm_vision_loss, mm_text_loss, align_loss)
            else:
                if 'vision' in models:
                    vision_outputs = models['vision']['model'](vision_data.to(device))
                    predictions = vision_outputs['cls']
                    loss = models['vision']['criterion'](predictions, label.to(device))
                else:
                    text_data_input = models['text']['tokenizer'](text_data, padding='longest', truncation=True, return_tensors='pt')['input_ids'].to(device)
                    text_outputs = models['text']['model'](text_data_input.to(device))
                    predictions = text_outputs['cls']
                    loss = models['text']['criterion'](predictions, label.to(device))

            train_metrices.update(label, predictions)

            for key in models:
                models[key]['optimizer'].zero_grad()

            loss.backward()

            for key in models:
                models[key]['optimizer'].step()

            for key in models:
                models[key]['scheduler'].step()

            epoch_loss += loss.item()
            
            tepoch.set_postfix({
                'Train Loss': f"{epoch_loss / (iter + 1):.4f}",
                'Train Accuracy': f"{train_metrices.compute(is_batch=True)['accuracy']:.4f}" 
            })
        train_history = train_metrices.compute()
        train_history['loss'] = epoch_loss / len(loader)
    return train_history 

def evaluate(loader, models, evaluation_metrices, device):
    epoch_loss = 0
    for key in models:
        models[key]['model'].eval()
    with torch.no_grad():
        with tqdm(loader, unit="batch", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=False) as tepoch:               
            train_ftext = f"   └─ Evaluation"              
            tepoch.set_description(f"{train_ftext:18}")       
            for iter, data in enumerate(tepoch):

                vision_data, text_data, label = data[0], data[1], data[2]

                if 'multimodal' in models:
                    predictions, mm_vision_loss, mm_text_loss, align_loss = models['multimodal']['model'](models, vision_data, text_data, label, device)
                    loss = models['multimodal']['criterion'](predictions, label.to(device), mm_vision_loss, mm_text_loss, align_loss)
                else:
                    if 'vision' in models:
                        vision_outputs = models['vision']['model'](vision_data.to(device))
                        predictions = vision_outputs['cls']
                        loss = models['vision']['criterion'](predictions, label.to(device))
                    else:
                        text_data_input = models['text']['tokenizer'](text_data, padding='longest', truncation=True, return_tensors='pt')['input_ids'].to(device)
                        text_outputs = models['text']['model'](text_data_input.to(device))
                        predictions = text_outputs['cls']
                        loss = models['text']['criterion'](predictions, label.to(device))

                evaluation_metrices.update(label, predictions)

                epoch_loss += loss.item()
                
                tepoch.set_postfix({
                    'Test Loss': f"{epoch_loss / (iter + 1):.4f}",
                    'Test Accuracy': f"{evaluation_metrices.compute(is_batch=True)['accuracy']:.4f}"
                })

    evaluation_history = evaluation_metrices.compute()
    evaluation_history['loss'] = epoch_loss / len(loader)
        
    return evaluation_history 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--save_dir', type=str, default='./dsma/saved')
    parser.add_argument('--seed', default=61, type=int, help='Random seed for training initialization.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

    parser.add_argument('--train_type', type=str, default='multimodal',
                        choices=['multimodal', 'vision', 'text'],
                        help='Which branch to train.')

    parser.add_argument('--vision_model', type=str, default='google/vit-base-patch16-224',
                        help='HuggingFace vision backbone (e.g., google/vit-base-patch16-224, '
                             'google/vit-large-patch16-224).')
    parser.add_argument('--text_model', type=str, default='emilyalsentzer/Bio_ClinicalBERT',
                        help='HuggingFace text backbone (e.g., emilyalsentzer/Bio_ClinicalBERT, '
                             'medicalai/ClinicalBERT, bert-base-uncased).')
    parser.add_argument('--dataset', type=str, default='../datasets')
    parser.add_argument('--include_dirs', type=str, nargs='*',
                        default=['2024 - BrEaST', '2026 - BUS-CoT'],
                        help='Dataset directories to include (e.g., "2020 - BUSI", '
                             '"2023 - BLUI", "2024 - BUS-UCLM", "2024 - BUSBRA").')
    parser.add_argument('--output_dim', type=int, default=3,
                        help='Number of output classes (2 for binary, 3 if "normal" is included).')
    parser.add_argument('--class_names', type=str, nargs='*',
                        default=['benign', 'malignant', 'normal'],
                        help='Class names matching output_dim.')
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--vision_lr', type=float, default=1e-5)
    parser.add_argument('--vision_weight_decay', type=float, default=5e-6)
    parser.add_argument('--text_lr', type=float, default=1e-5)
    parser.add_argument('--text_weight_decay', type=float, default=5e-6)
    parser.add_argument('--multimodal_lr', type=float, default=1e-5)
    parser.add_argument('--multimodal_weight_decay', type=float, default=5e-6)
    parser.add_argument('--fmap', type=int, default=768, help='Feature map dimension (768 for Base, 1024 for Large models)')
    parser.add_argument('--multimodal_fusion', type=str, default='product')
    parser.add_argument('--multilevel_fusion', type=str, default='concat')
    parser.add_argument('--fusion_stages', type=str, default='both',
                        choices=['lca_only', 'gaa_only', 'ccm_only', 'mha_only', 'both'],
                        help='Which fusion stages to use: lca_only (Stage I: Local Correlation Alignment), '
                             'gaa_only (Stage II: Global Attention Alignment), or both (default). '
                             'The legacy aliases ccm_only/mha_only are still accepted for backward compatibility.')
    parser.add_argument('--json_file', type=str, default='dataset.json', help='Name of the JSON dataset file to read')
    
    args = parser.parse_args()

    # Name the folder beautifully instead of just timestamps
    domain_tags = "_".join([d.split(' - ')[-1].replace(' ', '') for d in args.include_dirs])
    fusion_tag = f"_{args.fusion_stages}" if args.fusion_stages != 'both' else ""
    unique_folder = f'{args.train_type}_{args.vision_model.split("/")[-1]}_{args.text_model.split("/")[-1]}_{domain_tags}{fusion_tag}_{time.strftime("%m%d_%H%M")}'
    save_path_root = os.path.join(args.save_dir, unique_folder)
    os.makedirs(save_path_root, exist_ok=True)

    set_seed(args.seed)
    device = torch.device('cuda', args.gpu)

    global_metrices_train = MetricsHandler(args.output_dim)

    kfold_loader = MultiModalDataLoader(args, verbose=False)
    # Get the data and labels directly if you want to inspect
    dict_data, _ = ReadDatasets(args, include_dirs=args.include_dirs)
    all_train_history = []
    best_fold_predictions = {}
    
    with tqdm(kfold_loader, unit="fold", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tfold:
        for fold, (train_loader, test_loader) in enumerate(tfold):
            fold_history = []            
            fold_ftext = f"Fold {(fold+1)}|{len(kfold_loader)}"
            tfold.set_description(f"{fold_ftext:18}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.train_type == 'vision':
                models = {'vision':get_vision_model(args)}
            elif args.train_type == 'text':
                models = {'text':get_text_model(args)} 
            else:                 
                models = {'multimodal':get_multimodal_model(args),
                        'vision':get_vision_model(args),
                        'text':get_text_model(args)}                
            
            train_metrices = MetricsHandler(args.output_dim)
            evaluation_metrices = MetricsHandler(args.output_dim)

            best_test_acc = -float('inf')
            best_train_acc = -float('inf')

            with tqdm(range(1, args.epoch+1), unit="epoch", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=False) as tepoch:            
                for epoch in tepoch:
                    epoch_ftext = f"└─ Epoch {epoch:2}|{args.epoch}"
                    tepoch.set_description(f"{epoch_ftext:18}")

                    train_history = train(train_loader, 
                                        models, 
                                        train_metrices, 
                                        device)
                    
                    evaluation_history = evaluate(test_loader,
                                                models, 
                                                evaluation_metrices, 
                                                device)                 
                    
                    fold_history.append({'train':train_history, 'evaluation':evaluation_history})

                    tepoch.set_postfix({
                            'Train Loss': f"{train_history['loss']:.4f}",
                            'Test Loss': f"{evaluation_history['loss']:.4f}",
                            'Train Accuracy': f"{train_history['accuracy']:.4f}",
                            'Test Accuracy': f"{evaluation_history['accuracy']:.4f}",
                        })

                    if evaluation_history['accuracy'] >= best_test_acc:
                        if train_history['accuracy'] > best_train_acc:
                            best_test_acc = evaluation_history['accuracy'] 
                            best_train_acc = train_history['accuracy']                      
                            tfold.set_postfix({
                                'Train Loss': f"{train_history['loss']:.4f}",
                                'Test Loss': f"{evaluation_history['loss']:.4f}",
                                'Train Accuracy': f"{train_history['accuracy']:.4f}",
                                'Test Accuracy': f"{evaluation_history['accuracy']:.4f}",
                            })
                            best_fold_predictions[f'fold{fold+1}'] = {'y_preds':evaluation_metrices.y_preds, 
                                                                    'y_trues':evaluation_metrices.y_trues,
                                                                    'y_preds_proba':evaluation_metrices.y_preds_proba}
                            
                            torch.save({key:models[key]['model'].state_dict() for key in models}, 
                                       os.path.join(save_path_root, f'model_fold{fold+1}_best.pth'))
                                                
                    train_metrices.reset()
                    evaluation_metrices.reset()
            all_train_history.append(fold_history)
            global_metrices_train.y_preds += best_fold_predictions[f'fold{fold+1}']['y_preds']
            global_metrices_train.y_trues += best_fold_predictions[f'fold{fold+1}']['y_trues']
            global_metrices_train.y_preds_proba += best_fold_predictions[f'fold{fold+1}']['y_preds_proba']
            print(f"\nFold {fold+1} best: Train ACC: {best_train_acc:.4f} | Test ACC: {best_test_acc:.4f}")

        
        global_history = global_metrices_train.compute()

        # Save standard global history json
        with open(os.path.join(save_path_root, "global_metrics.json"), 'w') as f:
            json.dump({k: v for k, v in global_history.items() if isinstance(v, (int, float, str, dict))}, f, indent=4)

        torch.save({'folds_history':all_train_history,
                    'folds_predictions':best_fold_predictions,
                    'global_history':global_history}, 
                    os.path.join(save_path_root, "model_data.pth"))
        
        # --- NEW: Automated Publication-Ready Reporting ---
        print(f"\n[Reporting] Generating publication-ready charts in {save_path_root}...")
        
        # Calculate summary statistics for basic formatting
        global_history['loss'] = np.mean([f[-1]['evaluation']['loss'] for f in all_train_history]) # Approx final val loss
        
        # A) Statistical Summary (CSV/JSON)
        save_statistical_summary(global_history, save_path_root)
        
        # B) Confusion Matrix
        plot_confusion_matrix(global_metrices_train.y_trues, global_metrices_train.y_preds, args.class_names, save_path_root)
        
        # C) ROC AUC Curve
        plot_roc_auc(global_metrices_train.y_trues, global_metrices_train.y_preds_proba, args.class_names, save_path_root)
        
        # D) Learning Curves (Plot and CSV)
        plot_learning_curves(all_train_history, save_path_root)
        
        # E) Grad-CAM Visualizations (Using final Fold's models and test set)
        print(f"[Reporting] Generating Grad-CAM overlays for 10 samples...")
        fuse_model = models.get('multimodal', models.get('vision'))['model']
        save_gradcam_samples(test_loader, fuse_model, device, args.class_names, save_path_root, num_samples=10)
        
        print("[Reporting] Done! Check the saved folder for CSVs and PNGs.")
        
        # --- NEW: Independent Test Set Evaluation (Ensemble of K-Folds) ---
        print(f"\n[Testing] Evaluating ensemble of fold models on independent test set: dataset_test.json")
        try:
            args.json_file = 'dataset_test.json'
            test_only_loader = MultiModalDataLoader(args, fold=0, verbose=False)
            
            ensemble_y_preds_proba = None
            test_y_trues = None
            
            for fold in range(args.fold):
                checkpoint = torch.load(os.path.join(save_path_root, f'model_fold{fold+1}_best.pth'))
                
                if args.train_type == 'vision':
                    test_models = {'vision':get_vision_model(args)}
                elif args.train_type == 'text':
                    test_models = {'text':get_text_model(args)} 
                else:                 
                    test_models = {'multimodal':get_multimodal_model(args),
                                'vision':get_vision_model(args),
                                'text':get_text_model(args)}
                                
                for key in test_models:
                    if key in checkpoint:
                        test_models[key]['model'].load_state_dict(checkpoint[key])
                    test_models[key]['model'].to(device)
                    test_models[key]['model'].eval()
                    
                fold_evaluation_metrices = MetricsHandler(args.output_dim)
                evaluate(test_only_loader, test_models, fold_evaluation_metrices, device)
                
                if ensemble_y_preds_proba is None:
                    ensemble_y_preds_proba = np.array(fold_evaluation_metrices.y_preds_proba)
                    test_y_trues = fold_evaluation_metrices.y_trues
                else:
                    ensemble_y_preds_proba += np.array(fold_evaluation_metrices.y_preds_proba)
                    
            ensemble_y_preds_proba /= args.fold
            ensemble_y_preds = np.argmax(ensemble_y_preds_proba, axis=1)
            
            test_metrices = MetricsHandler(args.output_dim)
            test_metrices.y_preds = ensemble_y_preds.tolist()
            test_metrices.y_preds_proba = ensemble_y_preds_proba.tolist()
            test_metrices.y_trues = test_y_trues
            
            test_history = test_metrices.compute()
            
            test_save_root = os.path.join(save_path_root, "test_results_ensemble")
            os.makedirs(test_save_root, exist_ok=True)
            
            with open(os.path.join(test_save_root, "test_metrics.json"), 'w') as f:
                json.dump({k: v for k, v in test_history.items() if isinstance(v, (int, float, str, dict))}, f, indent=4)
                
            save_statistical_summary(test_history, test_save_root)
            plot_confusion_matrix(test_metrices.y_trues, test_metrices.y_preds, args.class_names, test_save_root)
            plot_roc_auc(test_metrices.y_trues, test_metrices.y_preds_proba, args.class_names, test_save_root)
            
            print(f"Independent test ACC: {test_history['accuracy']:.4f}")
            print(f"[Reporting] Independent test results saved in {test_save_root}")
            
        except Exception as e:
            print(f"[Error] Failed during independent test set evaluation: {e}")
