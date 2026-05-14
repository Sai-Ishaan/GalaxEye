## To stick to binary masks across all imgs
import numpy as np
import os
from tqdm import tqdm

def global_mask_cleaner(base_dir):
    splits = ['train', 'test', 'val']

    for split in splits:
        target_dir = os.path.join(base_dir, split)
        if not os.path.exists(target_dir):
            continue
        ##finding only target files
        files = [f for f in os.listdir(target_dir) if f.endswith('_target.npy')]
        print(f"\n Scanning {len(files)} files in {split} split")
        
        cleaned_count = 0
        for fname in tqdm(files):
            fpath = os.path.join(target_dir, fname) 
            mask = np.load(fpath)
            ##Checking for unique(non-binary) values(other than 0 and 1)
            unique_vals = np.unique(mask)
            is_pure_binary = np.all(np.isin(unique_vals, [0,1]))
            if not is_pure_binary:
                ##Forcing binary if values>0 becomes 1, else 0
                cleaned_mask = (mask > 0).astype(np.uint8)
                np.save(fpath, cleaned_mask)
                cleaned_count +=1
        print(f"Completed cleaning {cleaned_count} non-binary files in {split}!")

if __name__ == "__main__":
    global_mask_cleaner(r'E:\GalaxEye\processed_patches')           