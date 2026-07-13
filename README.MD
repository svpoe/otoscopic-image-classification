# Otoscopic Image Classification with Transfer Learning

## Overview

This project investigates transfer learning for multi-class classification of otoscopic images using an ImageNet-pretrained ResNet18.

The initial baseline uses the pretrained ResNet18 as a fixed feature extractor, with only the final classification layer trained on the otoscopic image dataset. Model performance is evaluated using a held-out test set, followed by Grad-CAM analysis to investigate which image regions influence model predictions.

## Objectives

- Build a reproducible image classification pipeline in PyTorch.
- Evaluate transfer learning using an ImageNet-pretrained ResNet18.
- Assess model performance using classification metrics and a confusion matrix.
- Use Grad-CAM to investigate model attention and potential shortcut learning.
- Analyze model failures and clinically relevant limitations.

## Dataset

The model was fine tuned on the Otoscopic Image Dataset:
https://www.kaggle.com/datasets/ucimachinelearning/otoscopic-image-dataset 

These data contain otsocopic images from five classes of medical conditions:
- Acute Otitis Media → 600 images
- Cerumen Impaction → 600 images
- Chronic Otitis Media → 600 images
- Myringosclerosis → 600 images
- Normal (Healthy cases) → 600 images

The data were divided into:

- 70% training
- 15% validation
- 15% testing

Stratified splitting was used to preserve class distributions across the three subsets.


## Method

### Preprocessing

Images were:

- resized to 224 × 224 pixels;
- converted to PyTorch tensors;
- normalized using ImageNet mean and standard deviation.

### Model

The model is based on an ImageNet-pretrained ResNet18.

For the initial transfer-learning baseline:

- the pretrained convolutional backbone was frozen;
- the original ImageNet classification layer was replaced with a new fully connected layer for five-class classification;
- only the new classification layer was trained.

### Training

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Epochs: 10
- Model selection: lowest validation loss

## Results

### Validation Performance

The best model checkpoint was selected based on validation loss.

| Metric | Result |
| --- | --- |
| Best validation loss | 0.0975 |
| Validation accuracy at best-loss epoch | 97.11% |
| Best checkpoint epoch | 9 |

![alt text](image.png)
### Test Performance

To be added after final test-set evaluation.

Planned metrics:

- Accuracy
- Macro F1-score
- Per-class recall
- Confusion matrix

## Grad-CAM Analysis

Grad-CAM is used to investigate which image regions contribute to model predictions.
![alt text](image-1.png)

## Project Structure
