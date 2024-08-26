# Metadata Augmented Deep Neural Networks for Wild Animal Classification

This repository contains code for classifying camera trap images using various deep learning models and metadata fusion techniques. 

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [File Descriptions](#file-descriptions)
4. [License](#license)
5. [Contributing](#contributing)
6. [Contact](#contact)

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/camera-trap-classification.git
```
```
cd camera-trap-classification
```

## Files

### metadata_classifier.py
- Implements a metadata-only classifier for camera trap images
- Handles data loading, preprocessing, and model training
- Performs experiments with different feature combinations and class selections
- Evaluates model performance using confusion matrices and various metrics

### fusion_model_v2.py
- Defines fusion models that combine image and metadata features
- Includes implementations for late fusion, early fusion, and metadata-only models

### conventional_models.py
- Contains implementations of various conventional deep learning models
- Includes ResNet, VGG, AlexNet, Inception, EfficientNet, and Vision Transformer (ViT) architectures
- These models are adapted to work with both image and metadata inputs

### viltkamera_classifier.py
- Main script for training and evaluating camera trap image classifiers
- Supports multiple model architectures and fusion techniques
- Handles data loading, augmentation, and preprocessing
- Implements training loops, model evaluation, and result logging

### evaluate_metadata_models.py
- Script for evaluating the performance of metadata-only models
- Analyzes results from different feature combinations and class selections
- Generates visualizations and statistics to compare model performances

## Usage

To train and evaluate models, run the `viltkamera_classifier.py` script. You can modify the model configurations, data preprocessing, and evaluation metrics within the script.

For metadata-only experiments, use the `metadata_classifier.py` script. The `evaluate_metadata_models.py` script can be used to analyze the results of these experiments.

Note: Make sure to set the correct paths for your dataset and adjust hyperparameters as needed.