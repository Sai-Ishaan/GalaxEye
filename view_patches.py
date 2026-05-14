import numpy as np
import os
from matplotlib import pyplot as plt
import random

def visualise_random_patches(base_dir, split='train'):
    patch_dir = os.path.join(base_dir, split)
    if not os.path.exists(patch_dir):
        print(f"Path not found: {patch_dir}")
        return
    pre_patches = [f for f in os.listdir(patch_dir) if f.endswith('_pre.npy')]
    if not pre_patches:
        print(f"No patches found in {patch_dir}")
        return
    ##Randomly selecting a patch
    choice = random.choice(pre_patches)
    base_name = choice.replace('_pre.npy', '')

    print(f"Visualising :{base_name}")

    pre = np.load(os.path.join(patch_dir, f"{base_name}_pre.npy"))
    post = np.load(os.path.join(patch_dir, f"{base_name}_post.npy"))
    target = np.load(os.path.join(patch_dir, f"{base_name}_target.npy"))

    #Plotting
    fig, axes = plt.subplots(1,3, figsize=(15,5))
    if len(pre.shape) == 3:
        axes[0].imshow(pre) ##Showing pre (EO) map to grayscale if its single channel
    else:
        axes[0].imshow(pre, cmap='gray')
    axes[0].set_title(f"Pre-Event (EO)\n Range: {pre.min():.2f}-{pre.max():.2f}")

    axes[1].imshow(post, cmap="gray")
    axes[1].set_title(f"Post-Event(SAR Filtered) \n Range: {post.min():.2f}-{post.max():.2f}")
    
    axes[2].imshow(target, cmap='binary')
    axes[2].set_title("Target Mask(Change)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualise_random_patches('E:\\GalaxEye\\processed_patches', 'train')