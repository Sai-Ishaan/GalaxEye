import os
import numpy as np
from PIL import Image
from glob import glob

def audit_dataset_imbalance(mask_dir):
    mask_files = glob(os.path.join(mask_dir, "*.tif"))
    total_change_pixels = 0
    total_pixels = 0
    print(f"Auditing {len(mask_files)} mask files...")

    for f in mask_files:
        mask = np.array(Image.open(f)) #Using PIL to keep memory footprint low
        total_change_pixels += np.sum(mask == 1)
        total_pixels += mask.size

    no_change_pixels = total_pixels - total_change_pixels
    ratio = total_change_pixels / total_pixels

    print(f"Total Pixels: {total_pixels}")
    print(f"Change Pixels(1): {total_change_pixels} ({ratio:.2%})")
    print(f"No Change Pixels(0): {no_change_pixels} ({1- ratio:.2%})")

    return ratio ##Note: if Ratio<0.05, switch to Weighted Cross Entropy Loss

audit_dataset_imbalance(r'E:\GalaxEye\train\target')
print("\n Auditing Test Dataset:")
audit_dataset_imbalance(r'E:\GalaxEye\fix\test\test\target')
print("\n Auditing Validation Dataset:")
audit_dataset_imbalance(r'E:\GalaxEye\fix\val\val\target')