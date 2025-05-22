# multimodal-cancer-grade-prediction

---

## Proteomics Model Evaluation and Visualization

### Overview  
This script evaluates SVM and Random Forest models on proteomics data, generates t-SNE visualizations of the dataset, and plots average protein expression profiles per tumor grade.

### Features

- Computes confusion matrices, classification reports, and accuracy scores for both models.
- Creates a t-SNE plot combining SMOTE-augmented training and test data.
- Plots average expression of selected proteins per tumor grade.
- Saves metrics to an Excel file and plots as PNG images in a `plots/` folder.

### Usage

1. Install dependencies:

    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
    ```

2. Run the proteomics evaluation script after training your models and preparing the data variables (`y_test`, `y_pred`, `X_train_resampled`, `X_selected`, etc.).

3. Outputs:
    - `model_metrics.xlsx` — Excel file with classification reports and confusion matrices.
    - `plots/` folder — contains the t-SNE and average protein expression plots.

---

# Histopathology Image Classification with ResNet50 Transfer Learning

This project uses TensorFlow and Keras to classify histopathology image tiles into tumor differentiation classes using transfer learning with a pretrained ResNet50 model. It includes data loading, augmentation, training, fine-tuning, evaluation, and feature extraction.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Preparation](#data-preparation)
- [Data Augmentation](#data-augmentation)
- [Model Architecture](#model-architecture)
- [Training & Fine-Tuning](#training--fine-tuning)
- [Evaluation](#evaluation)
- [Feature Extraction](#feature-extraction)
- [Dependencies](#dependencies)
- [Usage](#usage)

---

## Project Overview

- Input: Histopathology image tiles of size 224x224 pixels
- Classes: 3 tumor differentiation classes (e.g., `aca_bd`, `aca_md`, `aca_pd`)
- Model: Transfer learning using pretrained ResNet50 as a feature extractor
- Training: Initial training with frozen base model, followed by multiple fine-tuning phases with progressively more layers unfrozen
- Metrics: Accuracy, Precision, Recall; confusion matrix and classification report for evaluation
- Feature extraction: Extracts bottleneck features from the ResNet50 base model for downstream tasks

---

## Data Preparation

- Dataset split into 70% training and 30% temporary (validation + test) using `tf.keras.utils.image_dataset_from_directory` with `validation_split=0.3`.
- Temporary set further split manually or by custom logic into validation and test sets (not shown in code snippet).
- Labels are one-hot encoded (`label_mode='categorical'`).

---

## Data Augmentation

Training data augmentation pipeline applies:

- Random horizontal flips
- Small random rotations (±5%)
- Zoom (±5%)
- Brightness and contrast adjustments (±5%)
- Gaussian noise addition (stddev=0.01)

Augmentation is applied only on the training dataset.

---

## Model Architecture

- Base model: `ResNet50` pretrained on ImageNet without the top classification head (`include_top=False`).
- Base model layers frozen initially.
- Additional layers added on top:
  - BatchNormalization
  - GlobalAveragePooling2D
  - Dense(256, ReLU) + Dropout(0.5)
  - Dense(128, ReLU)
  - Output Dense layer with 3 units and softmax activation

---

## Training & Fine-Tuning

1. **Initial training** with frozen ResNet base for 25 epochs.
2. **Multiple fine-tuning phases**:
   - Unfreeze last 5 layers and train with lower learning rate.
   - Progressively unfreeze more layers (last 10, 15, 20 layers) for further fine-tuning.
   - BatchNormalization layers remain frozen during fine-tuning to stabilize training.
3. Each fine-tuning phase trains for additional epochs, continuing from previous training.

---

## Evaluation

- Predict on test dataset.
- Generate confusion matrix plot with seaborn.
- Print classification report including precision, recall, and F1-score for each class.

---

## Feature Extraction

- Extracts features from the ResNet50 base model using a separate feature extraction model.
- Features are pooled with GlobalAveragePooling2D.
- Extracted features saved as `.npy` files for train and validation datasets.

---

## Dependencies

```bash
pip install tensorflow matplotlib seaborn scikit-learn numpy
```

## Vision Transformer (ViT) Classifier for Histopathology Images 

This repository contains a PyTorch-based training pipeline for classifying histopathology images using a pretrained Vision Transformer (ViT). The model distinguishes between three tumor differentiation classes: `aca_bd`, `aca_md`, and `aca_pd`.

### 🧪 Project Summary

- **Model:** ViT-B/16 (`vit_base_patch16_224`)
- **Task:** Multiclass classification (3 classes)
- **Input:** Histopathology image tiles (224x224)
- **Output:** Predicted tumor class
- **Data Format:** `ImageFolder` with class subfolders
- **Transforms:** Includes standard augmentations + Gaussian noise

### 🚀 Usage

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

