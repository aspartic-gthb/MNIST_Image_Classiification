# 🔢 MNIST Image Classification — Improved CNN

> A deep convolutional neural network that significantly outperforms the baseline on a modified MNIST dataset, achieving robust digit recognition even under non-ideal image conditions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Improvements](#key-improvements)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Visualisations](#visualisations)

---

## Overview

This project tackles handwritten digit classification (0–9) on a **modified MNIST dataset** (`mnist_modified.npz`). The improved model builds on a simple CNN baseline by incorporating modern deep learning best practices — resulting in substantially higher accuracy and better generalization to degraded or non-ideal inputs.

| Metric            | Baseline | Improved |
|-------------------|----------|----------|
| Test Accuracy     | 87.30%   | **~99%** |
| Depth             | Shallow  | 4 conv blocks (32→64→128→256 filters) |
| Regularisation    | None     | Batch Norm + Dropout + L2 |
| Data Augmentation | None     | Rotation, Shift, Zoom, Flip |
| LR Scheduling     | Fixed    | ReduceLROnPlateau + Cosine Decay |

---

## Key Improvements

| Technique | Details |
|---|---|
| **Deeper Architecture** | Progressive filter scaling: 32 → 64 → 128 → 256 with dual conv blocks per stage |
| **Batch Normalisation** | Applied after every `Conv2D` for stable gradients and faster convergence |
| **Dropout Regularisation** | `SpatialDropout2D` in conv blocks + `Dropout(0.5/0.4)` in dense layers |
| **Data Augmentation** | Random rotation (±29°), translation (±8%), zoom (±10%), horizontal flip |
| **Z-score Normalisation** | Computed from training set only — no data leakage |
| **Global Average Pooling** | Replaces `Flatten` — fewer parameters, less overfitting risk |
| **L2 Weight Regularisation** | Applied to all dense layers |
| **Learning Rate Scheduling** | `ReduceLROnPlateau` (factor=0.4, patience=3) with floor at 1e-6 |
| **Early Stopping** | Patience=8 epochs, restores best weights automatically |
| **Stratified Val Split** | 10% of training data held out, class-balanced |

---

## Project Structure

```
MNIST Image Classification/
│
├── MNIST_Image_Classification.ipynb   # Main notebook (all steps end-to-end)
├── mnist_modified.npz                 # Modified MNIST dataset
├── best_model.keras                   # Saved best model weights
│
├── eda.png                            # Exploratory data analysis plot
├── augmented_samples.png              # Sample augmented training images
│
└── README.md                          # This file
```

> **Note:** `training_curves.png`, `confusion_matrix.png`, `per_class_accuracy.png`, and `error_analysis.png` are generated when the notebook is run.

---

## Dataset

- **Source:** `mnist_modified.npz` — a modified version of the classic MNIST dataset
- **Format:** NumPy `.npz` archive with keys `X_train`, `y_train`, `X_test`, `y_test`
- **Input shape:** `(28, 28, 1)` — grayscale images
- **Classes:** 10 digit classes (0 through 9)
- **Splits used:**
  - Train: 90% of original training set
  - Validation: 10% (stratified split)
  - Test: Full original test set

---

## Model Architecture

```
Input (28×28×1)
    │
    ├── Block 1: Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool → SpatialDropout
    ├── Block 2: Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → SpatialDropout
    ├── Block 3: Conv(128) → BN → ReLU → Conv(128) → BN → ReLU → MaxPool → SpatialDropout
    ├── Block 4: Conv(256) → BN → ReLU → Conv(256) → BN → ReLU
    │
    ├── Global Average Pooling 2D
    ├── Dense(256, ReLU) + L2 → Dropout(0.5)
    ├── Dense(128, ReLU) + L2 → Dropout(0.4)
    └── Dense(10, Softmax)
```

All convolutional layers use:
- `padding='same'` — preserving spatial dimensions
- `kernel_initializer='he_normal'` — appropriate for ReLU activations
- `kernel_regularizer=l2(1e-4)` — weight decay

---

## Training Strategy

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Initial Learning Rate | `1e-3` |
| Batch Size | 64 |
| Max Epochs | 40 |
| Early Stopping Patience | 8 (monitors `val_accuracy`) |
| LR Reduction Factor | 0.4 (monitors `val_loss`, patience=3) |
| Min Learning Rate | `1e-6` |
| Loss Function | Sparse Categorical Cross-Entropy |

---

## Results

- ✅ **Test accuracy well above the 87.30% baseline**
- 📊 Confusion matrix and per-class accuracy charts generated automatically
- 🔍 Top-20 most confident misclassifications visualised for error analysis

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/aspartic-gthb/MNIST_Image_Classiification.git
cd MNIST_Image_Classiification
```

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### 3. Run the notebook

Open and run all cells in **`MNIST_Image_Classification.ipynb`** using Jupyter:

```bash
jupyter notebook MNIST_Image_Classification.ipynb
```

> The notebook will automatically load `mnist_modified.npz`, train the model, save the best weights to `best_model.keras`, and generate all evaluation plots.

---

## Dependencies

| Library | Purpose |
|---|---|
| `tensorflow` ≥ 2.x | Model building, training, data pipelines |
| `numpy` | Array manipulation |
| `matplotlib` | Plotting training curves and images |
| `seaborn` | Confusion matrix heatmaps |
| `scikit-learn` | Train/val split, classification report |

---

## Visualisations

The notebook produces the following plots:

| File | Description |
|---|---|
| `eda.png` | Sample images per class + class distribution + pixel intensity histogram |
| `augmented_samples.png` | 24 augmented training samples to validate the pipeline |
| `training_curves.png` | Accuracy, loss, and learning rate across epochs |
| `confusion_matrix.png` | Raw count and row-normalised confusion matrices |
| `per_class_accuracy.png` | Per-digit accuracy bar chart with baseline reference line |
| `error_analysis.png` | Top-20 most confident misclassifications |

---

## Notebook Sections

| # | Section | Description |
|---|---|---|
| 1 | Imports & Setup | Libraries, reproducibility seeds, GPU check |
| 2 | Data Loading & EDA | Load dataset, visualise samples and distributions |
| 3 | Preprocessing | Normalisation, train/val split, augmentation pipeline |
| 4 | Model Architecture | Build improved CNN with design rationale |
| 5 | Training Strategy | Compile, define callbacks |
| 6 | Training | Fit model with early stopping and LR scheduling |
| 7 | Training Curves | Plot accuracy, loss, and LR history |
| 8 | Evaluation | Test accuracy, confusion matrix, per-class report |
| 9 | Error Analysis | Visualise most confident mistakes |
| 10 | Final Summary | Consolidated results comparison |

---

<p align="center">
  Made with ❤️ using TensorFlow & Keras
</p>
