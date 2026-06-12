import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_gradcam_samples(dataloader, multimodal_model, device, class_names, save_dir, num_samples=10):
    """
    Save side-by-side sample visualizations during training.

    This helper produces a centered-Gaussian overlay alongside the original
    image as a quick visual sanity check. The full multimodal Grad-CAM
    pipeline (with explicit hooks into the fused architecture) is implemented
    in dsma/test_gradcam.py and should be used for the qualitative figures
    reported in the paper.
    """
    cam_dir = os.path.join(save_dir, "gradcam_samples")
    os.makedirs(cam_dir, exist_ok=True)
    
    multimodal_model.eval()
    count = 0
    
    for batch in dataloader:
        if len(batch) == 3:
            images, text, labels = batch
        else:
            images, labels = batch
            
        images = images.to(device)
        labels = labels.to(device)
        
        # Determine actual predictions
        with torch.no_grad():
            if len(batch) == 3:
                outputs = multimodal_model(images, text)
            else:
                outputs = multimodal_model(images)
            preds = torch.max(outputs, 1)[1]
            trues = torch.max(labels, 1)[1]
            
        # Draw and save samples
        for i in range(images.size(0)):
            if count >= num_samples:
                return
                
            img_tensor = images[i].cpu()
            
            # Denormalize using ImageNet statistics for display
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_disp = img_tensor * std + mean
            img_disp = torch.clamp(img_disp, 0, 1)
            img_np = img_disp.permute(1, 2, 0).numpy()

            # Centered Gaussian placeholder overlay (see module docstring;
            # use test_gradcam.py for the full Grad-CAM pipeline).
            y, x = np.ogrid[-1:1:224j, -1:1:224j]
            mask = np.exp(-(x**2 + y**2) * 5)
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam_result = heatmap * 0.4 + img_np * 0.6
            cam_result = np.clip(cam_result, 0, 1)
            
            true_cls = class_names[trues[i].item()]
            pred_cls = class_names[preds[i].item()]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(img_np)
            ax1.set_title(f"True: {true_cls}")
            ax1.axis('off')
            
            ax2.imshow(cam_result)
            ax2.set_title(f"Pred: {pred_cls}")
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(cam_dir, f"sample_{count:02d}_True_{true_cls}_Pred_{pred_cls}.png"), dpi=150)
            plt.close()
            
            count += 1
