import os
import glob
import numpy as np
import torch
import torch.nn.functional as Func
from tqdm import tqdm
from model import SiameseUNet

def run_test_inference(checkpoint_path="best_model_epoch_1.pth"):
    device = torch.device("cpu")  # Forcing CPU inference
    test_dir = r"E:\GalaxEye\processed_patches\test"
    output_dir = r"E:\GalaxEye\predictions_test"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading weights from checkpoint: {checkpoint_path}")
    model = SiameseUNet(num_classes=1) # reconstruct model architecture

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() ## freezing batch normalization layers

    pre_paths = sorted(glob.glob(os.path.join(test_dir, "*_pre.npy")))
    print(f"Indexed {len(pre_paths)} test patch triplets awaiting generation.")

    ##Inference generation Loop
    print("Generating binary change masks...")
    with torch.no_grad():
        ## Eliminates grad tracking history to protect system RAM
        for pre_path in tqdm(pre_paths, desc="Processing"):
            post_path = pre_path.replace("_pre.npy", "_post.npy")
            base_name = os.path.basename(pre_path).replace("_pre.npy", "")

# Shape formatting matching batch expectations: Batch=1, Channel=1, height=256, width=256
            x1 = torch.from_numpy(np.load(pre_path)).float().unsqueeze(0).unsqueeze(0)
            x2 = torch.from_numpy(np.load(post_path)).float().unsqueeze(0).unsqueeze(0)
            logits = model(x1, x2) ## Compute predictions
            probabilities = torch.sigmoid(logits).squeeze().numpy()
            binary_mask = (probabilities > 0.5).astype(np.uint8)

            #Serialization to disk
            output_path = os.path.join(output_dir, f"{base_name}_predicted_change.npy")
            np.save(output_path, binary_mask)

    print(f"Inference complete! Binary arrays exported to: {output_dir}")

if __name__ == "__main__":
    run_test_inference("best_model_epoch_1.pth")