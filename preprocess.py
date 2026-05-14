import numpy as np 
import pandas as pd
import os
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance
import rasterio
from tqdm import tqdm
import random

def lee_filter(img, size=5):
    img = img.astype(np.float32)
    ### Std Lee Filter for SAR Speckle Noise reduction
    img_mean = uniform_filter(img, (size,size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance/(img_variance + overall_variance + 1e-10)
    output = img_mean + img_weights * (img - img_mean)
    return output

def create_patches(image, patch_size=256):##Slices a large img into patches
    patches = []
    height, width = image.shape[:2]
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches

def normalize_image(img): ##Normalizes pixel values to [0,1]
    ##Standardising image pixel values to [0,1] for the SNN
        img_min, img_max = img.min(), img.max()
        if img_max - img_min == 0:
            return img
        return (img - img_min) / (img_max - img_min)

def process_dataset(base_path, split_name, custom_path=None):
    root = custom_path if custom_path else os.path.join(base_path,split_name)
    pre_dir = os.path.join(root, 'pre-event')
    post_dir = os.path.join(root, 'post-event')
    target_dir = os.path.join(root, 'target')

    output_base = f"E:\\GalaxEye\\processed_patches\\{split_name}"

    os.makedirs(output_base, exist_ok=True)

    if not os.path.exists(target_dir):
        print(f"Path not found: {target_dir}")
        return
    
    ##filenames = os.listdir(target_dir)
    target_files = [f for f in os.listdir(target_dir) if f.endswith('.tif')]
    
    for fname in tqdm(target_files, desc=f"Processing {split_name}"):
        base_id = fname.split('_building_damage')[0]
        pre_file = next((f for f in os.listdir(pre_dir) if base_id in f), None)
        post_file = next((f for f in os.listdir(post_dir) if base_id in f), None)

        if not pre_file or not post_file:
            print(f"Skipping {base_id}: Missing pair in pre/post folders")
            continue
        
        try:
            with rasterio.open(os.path.join(pre_dir, pre_file)) as s: pre_img = s.read(1)
            with rasterio.open(os.path.join(post_dir, post_file)) as s: post_img = s.read(1)
            with rasterio.open(os.path.join(target_dir, fname)) as s: target_img = s.read(1)

            # Filtering and normalization
            post_filtered = lee_filter(post_img)
            pre_norm = normalize_image(pre_img.astype(np.float32))
            post_norm = normalize_image(post_filtered)

            # Creating patches
            pre_patches = create_patches(pre_norm)
            post_patches = create_patches(post_norm)
            target_patches = create_patches(target_img)

            # Saving
            for idx, (p1, p2, t) in enumerate(zip(pre_patches, post_patches, target_patches)):
                patch_id = f"{base_id}_p{idx}"
                np.save(os.path.join(output_base, f"{patch_id}_pre.npy"), p1)
                np.save(os.path.join(output_base, f"{patch_id}_post.npy"), p2)
                np.save(os.path.join(output_base, f"{patch_id}_target.npy"), t)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

def check_mask_values(base_dir):
    patch_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(patch_dir):
        print(f"Path not found:{patch_dir}")
        return
    target_patches = [f for f in os.listdir(patch_dir) if f.endswith('_target.npy')]
    if not target_patches:
        print("No target patches found")
        return 
    print(f"Checking 5 random masks from {len(target_patches)} files....")

    for _ in range(5):
        fname = random.choice(target_patches)
        mask = np.load(os.path.join(patch_dir, fname))
        unique_vals = np.unique(mask)
        print(f"File: {fname} | Unique Values: {unique_vals}")

        ##Heuristic check
        if len(unique_vals) >2:
            print("WARNING: More than 2 values found. Is this binary?")

if __name__ == "__main__":
    #Std train path
    process_dataset('E:\\GalaxEye', 'train')

    process_dataset(base_path='E:\\GalaxEye', split_name='test', custom_path='E:\\GalaxEye\\fix\\test\\test')
    process_dataset(base_path='E:\\GalaxEye', split_name='val', custom_path='E:\\GalaxEye\\fix\\val\\val')
    check_mask_values('E:\\GalaxEye\\processed_patches')