### a cpu bound training loop 
# implementing Jaccard Similarity (IoU) directly during validation to track accuracy of our change predictions
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from model import SiameseUNet
import gc # Garbage collector

##Here we are isolating system resources explicitly to avoid unoptimized multi-threading overhead
os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"   # Limit MKL threads

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forcing CPU usage instead og GPU
torch.set_num_threads(4)  # Limit PyTorch threads
## Creating a lightweight vectorised CPU data feeder
##Raw Numpy patches loaded sequentially to optimise particular memory footprint

class SiamesePatchDataset(Dataset): ## creating a custom dataset class to load preprocessed patches
    def __init__(self, patch_dir):
        self.pre_paths = sorted(glob.glob(os.path.join(patch_dir, "*_pre.npy")))
        if len(self.pre_paths) == 0:
            raise FileNotFoundError(f"No patches found in path: {patch_dir}")
        print(f"Dataset successfully indexed {len(self.pre_paths)}")
    def __len__(self):
        return len(self.pre_paths)

    def __getitem__(self, idx):
        pre_path = self.pre_paths[idx]
        post_path = pre_path.replace("_pre.npy", "_post.npy")
        target_path = pre_path.replace("_pre.npy", "_target.npy")

        ##Loading as float32 arrat and casting to torch Tensors
        #unsqueeze(0) helps to convert 256x256 into structural tensor format( as in channel=1, height=256, width=256)
        x1 = torch.from_numpy(np.load(pre_path)).float().unsqueeze(0)
        x2 = torch.from_numpy(np.load(post_path)).float().unsqueeze(0)
        y = torch.from_numpy(np.load(target_path)).float().unsqueeze(0)

        return x1,x2, y

#Logic for computing jaccard index/IoU exclusively for Change(1) class.
def calculate_jaccard_index(preds, target, smooth=1e-16):
    probs = torch.sigmoid(preds) 
    preds_binary = (probs>0.5).float() #converting real raw logits to hard binary (0 or 1) o/p
    intersection = (preds_binary * target).sum()
    total = preds_binary.sum() + target.sum()
    union = total - intersection

    jaccard = (intersection + smooth) / (union + smooth) # Here, tensor states across arrays natively in memory are evaluated
    return jaccard.item()

def run_training():
    device = torch.device("cpu") ## resource allocation
    print(f"Initialising CPU Training Loop Engine")

    train_dir = r"E:\GalaxEye\processed_patches\train"
    val_dir = r"E:\GalaxEye\processed_patches\val"

    BATCH_SIZE = 32 
    print(f"Loading training datasets WITH batch size: {BATCH_SIZE}")
    train_set = SiamesePatchDataset(train_dir)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True) ## Using batch size of 4 to balance memory and convergence
    
    print("Loading Validation Dataset")
    val_set = SiamesePatchDataset(val_dir)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    model = SiameseUNet(num_classes=1).to(device)

    ##Imbalance adjustment: We've a 13.43% chance of change pixels. Hence we translate that to a ~6.4 +ve weighting factor
    pos_weight_tensor = torch.tensor([6.4]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    ##Std Adam Learning rate for safe feature migration without weight explosion
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_jaccard = 0.0 ##CheckPt variable tracking
    epochs = 3
    print("Starting Training Cycle: ")
    print("\n*****")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_jaccard = 0.0

        ## Wrapped with progress bar for tracking throughput speed
        train_bar  = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, (eo, sar, target) in enumerate(train_bar):
            optimizer.zero_grad()

            #Fwd computation
            predictions = model(eo, sar)
            loss = criterion(predictions, target)

            loss.backward() #Error backwrd propagation step
            optimizer.step() #Weight update step

            train_loss += loss.item()
            batch_jaccard = calculate_jaccard_index(predictions.detach(), target)
            train_jaccard += batch_jaccard

            ##Live tracking display update every 20 batches
            if i%50 == 0:
                train_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Jaccard": f"{batch_jaccard:.4f}"})
        avg_train_loss = train_loss / len(train_loader)
        avg_train_jacc = train_jaccard / len(train_loader)

        ##Validation phase after each epoch
        model.eval()
        val_loss = 0.0
        val_jaccard = 0.0

        with torch.no_grad(): #Disabling tracking graphs to preserve RAM footprint
                val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for eo, sar, target in val_bar:
                    predictions = model(eo, sar)
                    loss = criterion(predictions, target)
                    val_loss += loss.item()
                    val_jaccard += calculate_jaccard_index(predictions, target)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_jacc = val_jaccard/ len(val_loader)

        print(f"\n[Epoch {epoch+1} Metrics Overview Summary]")
        print(f" Training - Loss: {avg_train_loss:.4f} | Mean Jaccard (IoU): {avg_train_jacc:.4%}")
        print(f" Validation - Loss: {avg_val_loss:.4f} | Mean Jaccard (IoU): {avg_val_jacc:.4%}")

            ## Auto-weight Saving CheckPt logic
        if avg_val_jacc > best_val_jaccard:
            best_val_jaccard = avg_val_jacc
            checkpoint_path = f"best_model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_jaccard': best_val_jaccard,
                'val_loss': avg_val_loss
            }, checkpoint_path)
                
            print(f"Completed! Checkpoint successfully written out to: {checkpoint_path}\n")
        else:
            print("No validation metric breakthrough recorder in this epoch cycle")
        print("-"*50)

        gc.collect() ## Free up CPU Cache memory allocations manually

if __name__ == "__main__":
    run_training()                    

