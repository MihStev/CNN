# 🐾 Animal Image Classifier — CNN

> A multi-class image classification system built with Convolutional Neural Networks, trained to distinguish between **cats**, **dogs**, and **wild animals** with **99.02% test accuracy**.

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-99.02%25-brightgreen?style=for-the-badge)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Definition](#-problem-definition)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Strategy & Iterations](#-training-strategy--iterations)
- [Final Results](#-final-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Authors](#-authors)

---

## 🧠 Overview

This project was developed as part of the **Neural Networks (13E054NM)** course at the **School of Electrical Engineering, University of Belgrade**.

The goal was to design, train, and iteratively improve a CNN capable of classifying animal photographs into three semantically distinct categories. The final model achieves near-human-level performance and was built from scratch — no pretrained weights.

---

## 🎯 Problem Definition

**Task:** Multi-class Image Classification (3 classes)  
**Input:** Static RGB images  
**Output:** Probability distribution over 3 classes

| Class | Description |
|-------|-------------|
| 🐱 **Cat** | Domestic cats |
| 🐶 **Dog** | Domestic dogs of various breeds |
| 🦊 **Wild Animal** | Wolves, foxes, tigers, lions, leopards |

> **Key Challenge:** The *Wild Animal* class shares strong visual features with the other two classes (e.g., a wolf resembles a husky, a tiger resembles a domestic cat). This high **inter-class similarity** makes the problem non-trivial.

---

## 📊 Dataset

The dataset consists of ~**16,500 RGB images** split into three independent subsets:

| Split | Proportion | Purpose |
|-------|-----------|---------|
| **Training** | ~70% | Parameter optimization |
| **Validation** | ~15% | Hyperparameter tuning & Early Stopping |
| **Test** | ~15% | Final, unbiased evaluation |

### Class Balance

Images per class range from **5,238 to 5,653**, with a maximum class imbalance of **7.34%** — well within the safe threshold. As a result, no oversampling (SMOTE) or class weighting was required.

All images were originally **512×512px** and resized to **128×128px** during preprocessing. Pixel values were normalized from `[0, 255]` → `[0, 1]`.

---

## 🏗️ Model Architecture

### Common Settings Across All Iterations

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Hidden activation | **ReLU** | Efficient, solves vanishing gradient |
| Output activation | **Softmax** | Multi-class probability output |
| Loss function | **Sparse Categorical Crossentropy** | Integer-encoded labels |
| Optimizer | **Adam** | Adaptive learning rate per parameter |
| Batch size | **32** | Balance between speed and stability |
| Input resolution | **128×128** | Preserves discriminative features, reduces compute |

---

## 🔁 Training Strategy & Iterations

### Iteration 1 — Baseline Model

**Architecture:** 3 Conv blocks (16 → 32 → 64 filters) + MaxPooling + Dense(64)

| Metric | Value |
|--------|-------|
| Test Accuracy | **96.67%** |
| Errors (test set) | 75 / 2250 |

**Observations:**
- Training accuracy climbed to ~99% while validation plateaued at ~94% → clear **overfitting**
- Model was overconfident on wrong predictions (confidence = 1.00)
- Relied on coarse silhouette features (e.g., pointy ears → cat/wild)

---

### Iteration 2 — Regularization & Data Augmentation

Three techniques were introduced simultaneously to combat overfitting:

1. **Dropout (0.5)** — placed before the output layer to reduce co-adaptation of neurons
2. **Early Stopping** — monitors `val_loss`, restores best weights automatically
3. **Data Augmentation** — random horizontal flips, rotation (±36°), zoom (±20%), contrast adjustment

**Augmentation samples:**

| Original | Zoom In | Zoom Out | Rotation | Contrast |
|----------|---------|---------|---------|---------|
| ![original](https://via.placeholder.com/80x80/cccccc/666?text=orig) | ![zoom_in](https://via.placeholder.com/80x80/cccccc/666?text=z+in) | ... | ... | ... |

> *(Replace with actual augmentation preview images from `/docs/`)*

| Metric | Value |
|--------|-------|
| Test Accuracy | **96.76%** |
| Errors (test set) | 73 / 2250 |

**Observations:**
- The sharp early accuracy spike disappeared — training became smoother and more stable
- Wrong predictions dropped from confidence ~1.00 to ~0.96–0.99
- Model began attending to fur texture and muzzle shape rather than silhouette alone

---

### Iteration 3 — Final Model (Deeper Architecture)

**Goal:** Maximize the model's ability to distinguish the most subtle visual features (e.g., wolf vs. husky fur texture).

**Changes made:**
- Added a **4th Conv block with 256 filters**
- Increased Dense layer to **256 neurons**
- **Batch Normalization deliberately omitted** — prior experiments showed it slowed training without meaningful gains on this dataset
- **Learning rate slightly reduced** to compensate for removed normalization

---

## 🏆 Final Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **99.02%** |
| **Total Errors** | 22 / 2250 |

The training and validation accuracy curves converge and overlap at ~98%, with no oscillations in either accuracy or loss — a sign of a well-generalizing model.

### Classification Report

```
              precision    recall  f1-score   support

         cat       1.00      0.99      1.00       750
         dog       0.99      0.99      0.99       750
        wild       0.98      0.99      0.99       750

    accuracy                           0.99      2250
   macro avg       0.99      0.99      0.99      2250
weighted avg       0.99      0.99      0.99      2250
```

### Confusion Matrix

```
         Predicted
          cat   dog   wild
True cat [ 745    3     2 ]
True dog [   0  739    11 ]
True wild[   1    5   744 ]
```

> The remaining misclassifications are **genuinely ambiguous** — most would challenge even a human observer (e.g., wolf pups resembling husky puppies, foxes resembling cats).

### Accuracy vs. Loss Curves

| Training Curves (Final Model) |
|-------------------------------|
| Accuracy and loss curves show synchronized convergence of train/val metrics across 35 epochs with no signs of overfitting. |

> *(Add your plot images under `/docs/` and reference them here with `![](docs/accuracy_plot.png)`)*

---

## 📁 Project Structure

```
CNN/
├── src/
│   └── *.py              # Model definition, training, evaluation scripts
├── docs/
│   └── *.png / *.pdf     # Report, plots, confusion matrices
├── .gitignore
├── LICENSE               # Apache 2.0
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Run Training

```bash
cd src
python train.py
```

### Run Evaluation

```bash
python evaluate.py
```

> **Note:** Place the dataset in a `data/` directory with subdirectories `cat/`, `dog/`, `wild/` (or update the path in the scripts).

---

## 👨‍💻 Authors

| Name | Index |
|------|-------|
| **Dobrica Janković** | 2022/0010 |
| **Mihajlo Stevanović** | 2022/0315 |

**Mentor:** Marija Novičić  
**Course:** Neural Networks — 13E054NM  
**Institution:** School of Electrical Engineering, University of Belgrade

---

<div align="center">
  <sub>Built from scratch · No pretrained weights · Apache 2.0 License</sub>
</div>
