# Car Type Classification using MobileNetV2

## Overview

This project implements a deep learning-based car type classification model using transfer learning.
A pre-trained MobileNetV2 model is fine-tuned to classify car images into multiple categories.

The goal of this project is to build an efficient and accurate image classifier while applying practical deep learning techniques such as data augmentation, fine-tuning, and regularization.

---

## Model Architecture

* **Base Model**: MobileNetV2 (pre-trained on ImageNet)
* **Input Size**: 128 × 128 × 3
* **Transfer Learning Strategy**:

  * Freeze all layers
  * Unfreeze the last 20 layers for fine-tuning

### Custom Classification Head

* GlobalAveragePooling2D
* Dense (128, ReLU)
* BatchNormalization
* Dense (64, ReLU)
* Dropout (0.5)
* BatchNormalization
* Dense (Softmax)

---

## Dataset

* Directory-based dataset
* Automatically loaded using `ImageDataGenerator`
* Split:

  * Training: 80%
  * Validation: 20% (via `validation_split`)

### Note

Data augmentation is applied to both training and validation datasets.
This may slightly bias validation accuracy.
In a production environment, separating train/validation directories is recommended.

---

## Data Augmentation

* Rotation (±30°)
* Zoom (up to 30%)
* Horizontal Flip
* Width/Height Shift (±10%)

---

## Training Configuration

* **Optimizer**: Adam (learning rate = 1e-4)
* **Loss Function**: Categorical Crossentropy
* **Batch Size**: 64
* **Epochs**: 100

### Callbacks

* ModelCheckpoint (save best model)
* EarlyStopping (patience = 10)
* CSVLogger (training logs)

---

## 📈 Results

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 96.68% |
| Validation Accuracy | 92.54% |
| Training Loss       | 0.2134 |
| Validation Loss     | 0.3010 |

The model demonstrates strong generalization performance with minimal overfitting.

