# GESTURE-Net: Gesture Aware and Agnostic Deep Learning for Robotic Surgical Error Detection

This repository contains the source code, dataset preprocessing pipeline, model architecture, and evaluation framework for **GESTURE-Net**, a hybrid deep learning model designed to detect executional errors in robotic-assisted surgeries using the **JIGSAWS dataset**.

---

## Overview

Robotic-assisted surgery enables high-precision operations, but executional errors—such as improper needle handling or loss of tool visibility—remain a critical concern. This project proposes GESTURE-Net, a dual-pathway deep learning model combining gesture-aware and gesture-agnostic reasoning to robustly detect such errors from surgical video data.

---

## Objectives

- **Temporal modeling** using 1D CNN, GRU, and Self-Attention
- **Gesture-aware + gesture-agnostic fusion** for context-rich predictions
- **Feature-level augmentation** to improve model robustness
- **Evaluation** using metrics like F1, AUC, Jaccard, and ROC

---

## Dataset

- **JIGSAWS Dataset** (Johns Hopkins University)
  - Tasks used: *Suturing* and *Needle Passing*
  - Frame rate: 30 FPS (downsampled to 5 FPS)
  - Labels: Frame-level gesture label (G1–G15) and error label (0/1)

>  Note: Dataset is not included due to licensing. Please obtain it from the original source.

---

## Preprocessing

Video preprocessing and frame extraction:

# Extract 224x224 center-cropped frames at 5 FPS
cv2.VideoCapture + resizing + center crop



## Feature extraction:

ResNet-50 pretrained on ImageNet

Output: 2048-d feature vectors per frame

## Feature augmentation:
Gaussian noise
Random scaling
Temporal jittering

 # Model Architecture
## Spatial Encoding
ResNet-50 → 2048-d feature vectors per frame

## Temporal Modeling
- 1D CNN for local temporal context
- GRU for sequence learning
- Scaled Dot-Product Attention for long-range dependencies

## Dual Pathway
- Gesture-Agnostic Path: Only temporal features
- Gesture-Aware Path: Concatenates gesture embeddings with temporal features
- Dynamic fusion of both paths via softmax weighting

## Final Prediction
- Sigmoid output for binary classification of executional error

## Evaluation
- Evaluation Metrics:
- Accuracy
- Precision, Recall, F1-score
- AUC (ROC)
- Jaccard Index


## Visualizations
- Training/validation loss and accuracy curves
- Frame-wise prediction plots
- Dynamic weighting between gesture-aware/agnostic paths
- Confusion matrices and ROC curves

## Future Work
- Expand to more surgical tasks (e.g., knot-tying)
- Real-time feedback systems
- Integration with clinical workflows


# Acknowledgements
- Dataset: JIGSAWS - Johns Hopkins University
- model inspiration: Simonyan & Zisserman’s Two-Stream Net, Vaswani et al.'s Transformers


