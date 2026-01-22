# Swin Transformer on BraTS 2021 — Domain Shift Segmentation Project

## Overview
This project evaluates how well the **Swin Transformer** architecture adapts to a **domain shift** from natural RGB images to **multi‑modal medical imaging**.  
We use the **Swin‑T (Tiny)** variant (~29M parameters) and apply it to **brain tumor segmentation** on the **BraTS 2021** dataset.

The goal is to understand:
- how Swin behaves when shifted from 3‑channel natural images to 4‑channel MRI,
- whether its hierarchical windowed attention still performs well in a medical context,
- and what adaptations are required for multi‑modal 2D segmentation.

---

## Dataset: BraTS 2021
BraTS 2021 is a large medical imaging dataset focused on **glioblastoma segmentation**.

Each patient case includes:
- **Four MRI modalities**,  
- **An expert‑annotated voxel‑wise tumor mask**,  
- Over **1,200 patient cases**, making it suitable for training Swin‑T.

Segmentation requires the model to predict a **class label for every pixel**, producing an output mask that highlights the **extracted tumor regions**.

The dataset is available on Kaggle:  
**[BraTS 2021 Task 1 Dataset](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/data)**

Training a Swin‑T segmentation model on an **NVIDIA A40 GPU** is feasible within **<12 hours**.

---

## Project Structure
```
swin-brats/
│
├── README.md
├── requirements.txt
│
├── configs/                 # model + training configs (.yml)
│   ├── swin_tiny.yml
│   ├── train_config.yml
│   └── dataset_config.yml
│
├── data/
│   ├── raw/                 # downloaded BraTS .nii.gz files
│   ├── processed/           # preprocessed numpy arrays or slices
│   └── splits/              # train/val/test patient lists
│
├── src/
│   ├── models/              # Swin-T backbone + segmentation head
│   ├── data/                # dataset + preprocessing
│   ├── training/            # training loop, losses, metrics
│   ├── utils/               # logging, visualization, config loading
│   └── inference/           # prediction script
│
├── notebooks/               # EDA + sanity checks
└── experiments/             # logs + checkpoints
```

---

## Method
- Adapt Swin‑T patch embedding to accept **4 MRI channels** instead of 3 RGB channels.
- Train a **2D Swin‑UNet–style segmentation model** on axial slices.
- Evaluate performance using **Dice score** and **IoU**.
- Analyze robustness under domain shift and discuss architectural implications.

---

## Requirements
```
torch
torchvision
timm
numpy
nibabel
opencv-python
matplotlib
pyyaml
```

---

## Training
```
python src/training/train.py --config configs/train_config.yml
```

## Inference
```
python src/inference/predict.py --checkpoint path/to/model.pth
```

---

## Authors
**COMP 4360 — Dr. Cristopher Henry**  
**Group 5:**  
- Duc Do  
- Jordon Hong  
- Muhammad Safdar  

---

## Acknowledgements

### Swin Transformer Paper
Liu, Ze, et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.*  
Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

### Dataset
BraTS 2021 dataset hosted on Kaggle:  
**[https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/data](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/data)**