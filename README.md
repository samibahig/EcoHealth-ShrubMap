# ShrubMap Data Challenge

**High-Resolution Shrub Segmentation Pipeline for Wildfire Risk & Public Health**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)
[![Challenge](https://img.shields.io/badge/Shrubwise-Sprint%204-orange.svg)](https://wildfirecommons.org)

---

## Overview

ShrubMap is an end-to-end deep learning pipeline for high-resolution shrub segmentation from NAIP multispectral imagery across 6 ecologically diverse California sites. It integrates 12 complementary input channels (spectral indices + texture features) and achieves **IoU=0.8397** with a ResNet34-UNet architecture trained on 192×192 patches.

This pipeline is motivated by the public health consequences of wildfire smoke exposure. Accurate shrub maps yield better fuel load estimates, more precise PM2.5 projections, and ultimately more effective emergency preparedness for vulnerable communities.

**Best Performance (Sprint 4):**

| Model | IoU | F1 | Recall | Precision |
|---|---|---|---|---|
| ResNet34-UNet 192×192 run3 ★ | **0.8397** | 0.9055 | 0.9585 | 0.8579 |
| Ensemble 2×ResNet34 (run2+3) | 0.8320 | 0.9083 | 0.9607 | 0.8613 |

---

## Pipeline Architecture

```
NAIP 0.6m imagery (4 bands)
        ↓
Feature Engineering (12 channels)
R, G, B, NIR + NDVI, EVI, TGI, NDWI, Brightness, VARI, texture_var, texture_ent
        ↓
TLS LiDAR masks → reprojection EPSG:26910 → binary label maps (ground truth)
        ↓
Patch extraction 64×64 (stride=16, min_shrub=5%) → 6,566 patches
        ↓
Upsample 192×192 + normalization (p1–p99 per channel)
        ↓
Augmentation ×6 (MixUp, CutMix, Copy-Paste, geometric, spectral)
        ↓
ResNet34-UNet training (Dice+BCE, pos_weight=21, early stopping)
        ↓
Ensemble IoU-weighted soft voting
```

---

## Repository Structure

```
ShrubMap-Data-Challenge/
│
├── 01_data_preparation.ipynb          # Feature engineering, patch extraction, label alignment
├── 01b_data_preparation_extra.ipynb   # Augmentation ×6 (MixUp, CutMix, Copy-Paste)
├── 02_baseline_models.ipynb           # Random Forest, XGBoost, SVM baselines
├── 03_deep_learning.ipynb             # ResNet34/50-UNet training (Sprint 4)
├── 04_comparison_report.ipynb         # Model comparison, SHAP analysis, ensemble
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Study Sites

| Site | Biome | Masks | Split |
|---|---|---|---|
| Sedgwick Reserve | Oak savanna (300–500m) | 117 | Train |
| Calaveras Big Trees | Mixed conifer (1200–1500m) | 105 | Train |
| Independence Lake | Subalpine (2000m+) | 56 | Validation |
| DL Bliss | Riparian, Lake Tahoe | 27 | Test |
| Pacific Union College | Mediterranean coastal | 37 | Test |
| Shaver Lake | Mixed Sierra Nevada | 23 | Test |

**Total: 299 manually annotated TLS LiDAR masks**

---

## Environment Setup

### Option A — NRP JupyterHub (Recommended)

1. Log in at [datachallenge-jupyterhub.wildfirecommons.org](https://datachallenge-jupyterhub.wildfirecommons.org)
2. Configure server: 16 CPU cores, 64 GB RAM, GPU (automatically allocated by platform)
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run notebooks in order: `01` → `01b` → `02` → `03` → `04`

### Option B — Local (CPU only)

```bash
git clone https://github.com/samibahig/ShrubMap-Data-Challenge.git
cd ShrubMap-Data-Challenge
pip install -r requirements.txt
jupyter lab 01_data_preparation.ipynb
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data preparation
jupyter nbconvert --to notebook --execute 01_data_preparation.ipynb
jupyter nbconvert --to notebook --execute 01b_data_preparation_extra.ipynb

# 3. Run baseline models
jupyter nbconvert --to notebook --execute 02_baseline_models.ipynb

# 4. Run deep learning training (requires GPU, ~2-4 hours)
jupyter nbconvert --to notebook --execute 03_deep_learning.ipynb

# 5. Run comparison report
jupyter nbconvert --to notebook --execute 04_comparison_report.ipynb
```

Or simply open each notebook and run **Kernel → Restart & Run All**.

---

## Key Results

| Model | IoU | F1 | Recall | Precision | Epochs |
|---|---|---|---|---|---|
| NDVI Threshold (baseline) | 0.185 | 0.312 | 0.760 | — | — |
| Random Forest 128×128 + SMOTE | 0.571 | 0.728 | 0.827 | 0.651 | — |
| EfficientNet-B3 UNet | 0.684 | 0.806 | 0.945 | — | 176 |
| UNet 128×128 | 0.751 | 0.858 | 0.954 | 0.779 | 163 |
| UNet-ResNet50 128×128 | 0.757 | 0.844 | 0.957 | — | 124 |
| ResNet34-UNet 192×192 run3 ★ | **0.8397** | 0.9055 | 0.9585 | 0.8579 | 35 |
| Ensemble 2×ResNet34 (run2+3) | **0.8320** | 0.9083 | 0.9607 | 0.8613 | — |

**Literature benchmark:** Zhu et al. (2025) F1=0.789 — surpassed by our best model.

---

## Feature Engineering — 12 Channels

| # | Channel | Description |
|---|---|---|
| 1–4 | R, G, B, NIR | Raw NAIP bands |
| 5 | NDVI | Vegetation vigor |
| 6 | EVI | Enhanced vegetation (reduces soil noise) |
| 7 | TGI | Triangular greenness index |
| 8 | NDWI | Water index (excludes water bodies) |
| 9 | Brightness | Overall reflectance |
| 10 | VARI | Visible atmospherically resistant index |
| 11 | texture_var | Local NDVI variance (5×5 window) |
| 12 | texture_ent | NIR Shannon entropy (disk radius 3) |

**SHAP analysis:** `texture_ent` (0.171) and `texture_var` (0.099) are the most discriminative features.

---

## Public Health Motivation

Shrub mapping is a public health problem. The causal chain:

```
Shrub cover density → surface fuel load (kg/m²)
        ↓
Fuel load × ignition probability → fire intensity
        ↓
Fire intensity → PM2.5 emissions
        ↓
PM2.5 exposure → cardiopulmonary morbidity & mortality
```

In California, wildfires account for 50% of total primary PM2.5 emissions. A 30% increase in emergency visits at Rady's Children's Hospital was documented for each 10-unit PM2.5 increase from wildfire smoke (San Diego County). **Every false negative — every missed shrub patch — propagates directly to underestimated fire severity and inadequate public health preparedness.** This is why we prioritize Recall over Precision.

---

## Green AI Commitment

- **Transfer learning** — ResNet34/50 ImageNet pretrained weights (60–80% compute reduction vs scratch)
- **Early stopping** — patience=15, no wasted epochs
- **Patch-based training** — memory-efficient, no full GeoTIFF processing
- **Future:** CodeCarbon/CarbonTracker integration for CO2e reporting

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Sami Bahig** — AI Engineer & ML Researcher, MD MSc  
Wildfire Science & Technology Commons, University of California San Diego  
Shrubwise Data Challenge, Sprint 4 — April 2026
