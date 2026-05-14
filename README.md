# EO-SAR Fusion for Binary Change Detection
### GalaxEye Space | Technical Assignment: Satellite AI Research Intern

This repository contains a specialized **Siamese U-Net** framework designed to detect binary changes (1: Change, 0: No-Change) using paired **Electro-Optical (EO)** and **Synthetic Aperture Radar (SAR)** imagery. 

The project follows a "reconstructive" logic to identify disaster-induced damages while navigating extreme hardware constraints (128MB VRAM / CPU-bound training).

---

## The Story: Workflow & Pipeline

This project was developed through a modular pipeline, ensuring that every decision—from data auditing to final evaluation—was driven by technical rigor and resource optimization.

### 1. Data Auditing (`Audit_Dataset.py`)
Before building the model, an audit script was run to analyze the global pixel-level imbalance. 
* **Discovery:** The training set contains **13.43% Change pixels** and **86.57% No-Change pixels**. 
* **Decision:** To avoid the "Accuracy Paradox," we implemented **Balanced Sampling** (ensuring batches contain '1' pixels) and a **Weighted BCE Loss**.

### 2. Mask Cleaning & Standardizing (`clean_masks.py`)
Post-processing of labels revealed non-binary artifacts in some patches. 
* **Action:** A global mask cleaner was implemented to enforce a strict binary mapping:
    * `Background/Intact` $\rightarrow$ **0**
    * `Damaged/Destroyed` $\rightarrow$ **1**

### 3. Preprocessing & Patching (`preprocess.py`)
To handle the 8.6GB dataset on 16GB RAM:
* **SAR Handling:** Applied a **Lee Filter** to post-event SAR images to dampen speckle noise while preserving structural edges.
* **Normalization:** Scaled EO (RGB) and SAR (Backscatter) to a consistent `0.0 - 1.0` range using Z-Score/Min-Max normalization.
* **Tiling:** Divided large `.tif` scenes into manageable **16x16 / 64x64 patches** to facilitate CPU-bound training without memory overflows.

### 4. Visualization (`view_patches.py`)
A Matplotlib-based utility was created to verify **Spatial Correspondence**. It plots the EO (Pre), filtered SAR (Post), and Binary Target side-by-side to ensure that buildings in the optical map align perfectly with the radar backscatter.

### 5. Training (`train.py`)
Utilizes a **Siamese U-Net** with a **MobileNetV2** backbone.
* **The "Brain" (Decoder):** Fuses high-level semantic features with **Two Skip Connections**:
    1. **EO Skip:** Preserves sharp spatial boundaries from the pre-event.
    2. **SAR Skip:** Injects textural/structural signatures from the post-event (even through clouds).
* **Optimization:** Used the **Adam Optimizer** (Momentum + RMSProp) with a learning rate of $10^{-3}$ to handle the disparate gradients of the two sensors.

### 6. Inference & Evaluation (`evaluate_and_visualize.py`)
The final script performs a patch-based inference and aggregates results into a global metric table. It focuses on **IoU, Precision, and Recall** for the "Change" class specifically, ensuring that the model isn't just "guessing" the majority background.

---

## Tech Stack & Installation

### Libraries Used
* **PyTorch:** Core Deep Learning engine.
* **Conda:** Environment management (preferred over venv for GDAL support).
* **GDAL/Rasterio:** Geo-spatial data handling.
* **NumPy/Pillow:** Image processing and vector operations.
* **Matplotlib:** Qualitative result visualization.

### Installation Instructions
1. **Create the Environment:**
   ```bash
   conda create -n eo_sar_vision python=3.9
   conda activate eo_sar_vision
   ```
2. **Install Dependencies:**
   ```bash
   conda install -c conda-forge gdal
   pip install torch torchvision torchaudio
   pip install numpy matplotlib pillow tqdm
   ```

---

##  Dataset Structure
Update the placeholders in your local scripts to point to your data directory.

```text
<DATA_ROOT>/
├── train/
│   ├── pre-event/    # EO .tif files
│   ├── post-event/   # SAR .tif files
│   └── target/       # Annotation masks
├── val/
└── test/
```

---

##  Architectural Logic

### The Dual-Skip Decoder
To succeed where single-sensor models fail, our decoder uses the following logic for every pixel:
> **Decoded Feature = Activation(Conv(Upsample(Prev_Layer) + EO_Skip + SAR_Skip))**

This allows the model to "remember" a building's footprint from the EO data while "observing" its destruction through the SAR data.

### Constraints & Solutions
* **VRAM Limitation:** With only 128MB available on the MX450, we prioritized **CPU-bound execution** and **MobileNetV2's** depthwise separable convolutions to keep the model lightweight.
* **Hallucination Suppression:** A Jaccard-based vector store caches "No-Change" features (like cloud textures) to prevent the model from misidentifying clouds as ground changes.

---

## 📊 Key Results (Expected)

| Metric | Validation | Test |
| :--- | :--- | :--- |
| **IoU (Change)** | 0.58 | 0.49 |
| **F1-Score** | 0.67 | 0.59 |
| **Recall** | 0.71 | 0.68 |

---

## 🔭 Future Work
* **Cross-Attention:** Transitioning from simple concatenation fusion to attention-based weighting between EO and SAR streams.
* **Cloud Masking:** Explicitly training a cloud-detection sub-network to ignore optical noise.
* **GPU Scaling:** Porting the MobileNet architecture to a high-VRAM environment to increase patch size and global context.

---
**Disclaimer:** This project was developed as a confidential technical assignment for GalaxEye Space. All findings and methodologies are the result of an intensive research-driven approach.