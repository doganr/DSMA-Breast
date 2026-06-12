import os
import time
import json
import warnings
import logging
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# --- GRAD-CAM Imports ---
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models.vision_models import get_vision_model
from models.text_models import get_text_model
from models.multimodal_models import get_multimodal_model
from data.ultrasound_dataset import MultiModalDataLoader
from utils.metrics import MetricsHandler

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


# --- 1. Smart Multimodal Wrapper ---
class MultimodalGradCAMWrapper(torch.nn.Module):
    def __init__(self, models_dict, device):
        super(MultimodalGradCAMWrapper, self).__init__()

        # Register models with PyTorch so parameters are visible
        self.multimodal_model = models_dict['multimodal']['model']
        self.vision_model = models_dict['vision']['model']

        self.models_dict = models_dict
        self.device = device
        self.current_text_data = None
        self.current_label = None

    def set_current_batch(self, text_data, label):
        self.current_text_data = text_data
        self.current_label = label

    def forward(self, vision_input):
        # If GradCAM sends B=2 (to bypass squeeze error), we must duplicate our B=1 text data
        B = vision_input.size(0)

        # Prepare text data
        text_data_final = self.current_text_data

        # Case 1: Text data is a list/tuple of strings
        if isinstance(text_data_final, (list, tuple)):
            current_len = len(text_data_final)
            if current_len == 1 and B > 1:
                # Duplicate the single string to match batch size
                text_data_final = list(text_data_final) * B

        # Case 2: Text data is a dictionary of tensors (already tokenized)
        elif isinstance(text_data_final, dict):
            text_data_new = {}
            for k, v in text_data_final.items():
                if v.size(0) != B:
                    repeat_dims = [1] * v.dim()
                    repeat_dims[0] = B
                    text_data_new[k] = v.repeat(*repeat_dims)
                else:
                    text_data_new[k] = v
            text_data_final = text_data_new

        # Align Label dimensions
        if self.current_label is not None and self.current_label.size(0) != B:
            label_aligned = self.current_label.repeat(B, 1)
        else:
            label_aligned = self.current_label

        # Run the same forward pass but return logits instead of softmax probabilities
        # This is critical for Grad-CAM to get stable gradients.
        vision_data, label_aligned = vision_input.to(self.device), label_aligned.to(self.device)
        text_data_input = self.models_dict['text']['tokenizer'](text_data_final, padding='longest', truncation=True, return_tensors='pt')['input_ids'].to(self.device)

        vision_outputs = self.models_dict['vision']['model'](vision_data)
        text_outputs = self.models_dict['text']['model'](text_data_input)
        
        vision_embeddings = vision_outputs['embeddings']
        text_embeddings = text_outputs['embeddings']

        aligned_vision_embeddings, aligned_text_embeddings = self.multimodal_model.align_features(vision_embeddings, text_embeddings)

        vision_pool = torch.mean(aligned_vision_embeddings, dim=1)
        text_pool = torch.mean(aligned_text_embeddings, dim=1)

        semantic_vision_embeddings, semantic_text_embeddings = self.multimodal_model.calculate_semantic_embeddings(
            vision_pool, text_pool, vision_embeddings.size(0)
        )

        F_sem = self.multimodal_model.fusion(semantic_vision_embeddings, semantic_text_embeddings, self.multimodal_model.multimodal_fusion)
        F_pool = self.multimodal_model.fusion(vision_pool, text_pool, self.multimodal_model.multimodal_fusion)
        F_final = self.multimodal_model.fusion(F_sem, F_pool, self.multimodal_model.multilevel_fusion)

        hidden_fc = self.multimodal_model.dropout(self.multimodal_model.ReLu(self.multimodal_model.classification_fc1(F_final)))
        raw_logits = self.multimodal_model.classification_fc2(hidden_fc).squeeze()
        
        # Squeeze logic (ensure BxOut shape is preserved for a single batch)
        if raw_logits.dim() == 1 and vision_data.size(0) == 1:
            raw_logits = raw_logits.unsqueeze(0)
            
        return raw_logits


# --- 2. ViT Reshape Transform ---
def reshape_transform(tensor, height=14, width=14):
    # Depending on model (ViT vs DeiT with dist token), tensor sequence length might be 197 or 198
    if tensor.size(1) == 198: 
        # Has cls token and distillation token
        spatial_tokens = tensor[:, 2:, :]
    else:
        # Standard ViT (only cls token)
        spatial_tokens = tensor[:, 1:, :]
    
    result = spatial_tokens.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# --- 3. Image Saving (UPDATED: 3 PANELS) ---
def save_gradcam_image(img_tensor, grayscale_cam, save_path, filename):
    # Convert Tensor to NumPy and Un-normalize
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)

    # 1. Original Image (Left Panel)
    orig_img_uint8 = np.uint8(255 * img)

    # 2. Heatmap Only (Middle Panel)
    # Convert 0-1 range to 0-255 and apply colormap
    heatmap_uint8 = np.uint8(255 * grayscale_cam)
    # Use JET colormap (returns BGR)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    # Convert to RGB to match others
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # 3. Overlay (Right Panel)
    # Superimpose heatmap onto original image
    overlay = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    # Combine 3 images horizontally: [Original | Heatmap | Overlay]
    combined = np.hstack((orig_img_uint8, heatmap_colored_rgb, overlay))

    # Save to disk
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # OpenCV expects BGR, so convert RGB back to BGR before saving
    cv2.imwrite(os.path.join(save_path, filename), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization Script')
    parser.add_argument('--model_dir', type=str, default='./dsma/saved/20251225_093410/model_fold2_best.pth')
    parser.add_argument('--vision_model', type=str, default='google/vit-base-patch16-224')
    parser.add_argument('--text_model', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--evaluation', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='../datasets')
    parser.add_argument('--json_file', type=str, default='dataset_test.json')
    parser.add_argument('--include_dirs', type=str, nargs='*',
                        default=['2026 - BUS-CoT'],
                        help='Directories to include')
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--class_names', type=str, nargs='*',
                        default=['benign', 'malignant'],
                        help='Class names')
    # Default batch size set to 2 to avoid "squeeze" errors in single-sample batches
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--multimodal_fusion', type=str, default='product')
    parser.add_argument('--multilevel_fusion', type=str, default='concat')
    parser.add_argument('--fusion_stages', type=str, default='both')
    parser.add_argument('--num_workers', type=int, default=0)  # Set to 0 to prevent multiprocessing errors
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--target_label', type=int, default=0, help='0 for benign, 1 for malignant')

    args = parser.parse_args()
    device = torch.device('cuda', args.gpu)

    print('Checkpoint loading...')
    checkpoint = torch.load(args.model_dir, weights_only=True)
    print('Checkpoint loaded...')

    models = {'multimodal': get_multimodal_model(args, evaluation=True),
              'vision': get_vision_model(args, evaluation=True),
              'text': get_text_model(args, evaluation=True)}

    print('Models loading...')
    models['vision']['model'].load_state_dict(checkpoint['vision'])
    models['vision']['model'].to(device).eval()

    models['text']['model'].load_state_dict(checkpoint['text'])
    models['text']['model'].to(device).eval()

    models['multimodal']['model'].load_state_dict(checkpoint['multimodal'])
    models['multimodal']['model'].to(device).eval()
    print('Models loaded...')

    # Initialize Wrapper
    full_model_wrapper = MultimodalGradCAMWrapper(models, device)

    # Determine Target Layer
    try:
        # For HuggingFace ViT
        target_layers = [models['vision']['model'].vit.encoder.layer[-1].layernorm_after]
    except:
        try:
            # For Timm ViT
            target_layers = [models['vision']['model'].blocks[-1].norm1]
        except:
            print("WARNING: Target layer not found, trying the last layer available.")
            target_layers = [list(models['vision']['model'].modules())[-1]]

    # Initialize GradCAM
    cam = GradCAMPlusPlus(model=full_model_wrapper, target_layers=target_layers, reshape_transform=reshape_transform)

    test_loader = MultiModalDataLoader(args, fold=0, verbose=False)
    gradcam_save_dir = os.path.join(os.path.dirname(args.model_dir), "gradcam_samples")
    if not os.path.isdir(gradcam_save_dir):
        os.makedirs(gradcam_save_dir, exist_ok=True)
    saved_count = 0
    MAX_SAVE = 300

    with tqdm(test_loader, unit="batch", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=False) as tepoch:
        tepoch.set_description(f"Evaluating")
        for iter, data in enumerate(tepoch):
            # Input Preparation
            if isinstance(data[0], dict):
                vision_input = data[0]['pixel_values'].to(device)
            else:
                vision_input = data[0].to(device)

            # Text data from DataLoader (likely a tuple of strings)
            text_data = data[1]
            label = data[2].to(device)

            # Skip small batches to prevent squeeze errors
            if vision_input.size(0) < 2:
                continue

            # Load current batch into Wrapper
            full_model_wrapper.set_current_batch(text_data, label)

            # Get Predictions
            predictions = models['multimodal']['model'](models, vision_input, text_data, label, device)

            pred_classes = torch.argmax(predictions, dim=1)
            label_classes = torch.argmax(label, dim=1)

            for i in range(vision_input.size(0)):
                # Save only True Label based on args.target_label
                if label_classes[i] == args.target_label and pred_classes[i] == args.target_label and saved_count < MAX_SAVE:

                    single_input = vision_input[i].unsqueeze(0)
                    duplicated_input = single_input.repeat(2, 1, 1, 1)

                    # Handle Text Data (Tuple of strings -> Single string tuple)
                    if isinstance(text_data, (list, tuple)):
                        single_text_data = (text_data[i],)
                    else:
                        single_text_data = text_data

                    single_label = label[i].unsqueeze(0)

                    # Update Wrapper for single instance
                    full_model_wrapper.set_current_batch(single_text_data, single_label)

                    # Generate Grad-CAM
                    targets = [ClassifierOutputTarget(args.target_label)] * duplicated_input.size(0)
                    grayscale_cams = cam(input_tensor=duplicated_input, targets=targets)
                    grayscale_cam = grayscale_cams[0, :]

                    lbl_name = "malignant_TP" if args.target_label == 1 else "benign_TN"
                    filename = f"train_batch{iter}_img{i}_{lbl_name}.png"

                    # Save 3-Panel Image
                    save_gradcam_image(vision_input[i], grayscale_cam, gradcam_save_dir, filename)

                    saved_count += 1
                    print(f"Visual evidence saved: {filename}")

    if saved_count > 0:
        print(f"\n--- SUCCESS: {saved_count} Grad-CAM images saved to '{gradcam_save_dir}'! ---")
    else:
        print("\n--- WARNING: No True Negative Benign examples captured. ---")