import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from model import SiameseUNet

def evaluate_and_visualize(checkpoint_path="best_model_epoch_1.pth"):
    device = torch.device("cpu")
    test_dir = r"E:\GalaxEye\processed_patches\test"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return

    model = SiameseUNet(num_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 2. Select a random patch group from the Test Split
    pre_paths = sorted(glob.glob(os.path.join(test_dir, "*_pre.npy")))
    if not pre_paths:
        print("No test patches found.")
        return
        
    chosen_pre = random.choice(pre_paths)
    chosen_post = chosen_pre.replace("_pre.npy", "_post.npy")
    chosen_target = chosen_pre.replace("_pre.npy", "_target.npy")
    base_name = os.path.basename(chosen_pre).replace("_pre.npy", "")

    # 3. Load arrays and prepare tensors for model inference
    pre_np = np.load(chosen_pre)
    post_np = np.load(chosen_post)
    target_np = np.load(chosen_target)

    # Format shape to match batch expectations: (Batch=1, Channel=1, H=256, W=256)
    x1 = torch.from_numpy(pre_np).float().unsqueeze(0).unsqueeze(0)
    x2 = torch.from_numpy(post_np).float().unsqueeze(0).unsqueeze(0)

    # 4. Run Cross-Modal Inference
    with torch.no_grad():
        logits = model(x1, x2)
        probabilities = torch.sigmoid(logits).squeeze().numpy()
        predicted_mask = (probabilities > 0.5).astype(np.uint8)

    # 5. Plot the Quad-View Comparison Map
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(pre_np, cmap='gray')
    axes[0].set_title("Pre-Event (EO Ground Map)")
    
    axes[1].imshow(post_np, cmap='gray')
    axes[1].set_title("Post-Event (SAR Filtered)")
    
    axes[2].imshow(target_np, cmap='binary')
    axes[2].set_title("Ground Truth Mask")
    
    axes[3].imshow(predicted_mask, cmap='jet')
    axes[3].set_title("Model Predicted Change")
    
    for ax in axes:
        ax.axis('off')
        
    print(f"\nDisplaying Evaluation Output Matrix for Scene: {base_name}")
    print(f"Validation Jaccard achieved during checkpoint compile: {checkpoint.get('val_jaccard', 0.0):.4%}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_and_visualize("best_model_epoch_1.pth")    
