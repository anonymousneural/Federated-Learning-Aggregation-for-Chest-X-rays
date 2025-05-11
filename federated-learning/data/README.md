# Dataset Documentation for Federated Learning Project

## Overview
This document provides detailed information about the dataset used in the Federated Learning project. It includes data sources, preprocessing steps, and relevant statistics that are essential for understanding the dataset's structure and usage.

## Data Sources
The dataset utilized in this project is the Chest X-ray dataset, which is publicly available. It contains images of chest X-rays categorized into two classes: pneumonia and normal. The dataset can be accessed from the following source:
- [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## Dataset Structure
The dataset is organized into three main directories:
- `train`: Contains training images with subdirectories for each class (pneumonia and normal).
- `val`: Contains validation images with subdirectories for each class.
- `test`: Contains test images with subdirectories for each class.

The directory structure is as follows:
```
chest_xray/
├── train/
│   ├── pneumonia/
│   └── normal/
├── val/
│   ├── pneumonia/
│   └── normal/
└── test/
    ├── pneumonia/
    └── normal/
```

## Preprocessing Steps
The following preprocessing steps are applied to the dataset before training the models:
1. **Image Rescaling**: All images are rescaled to a range of [0, 1] by dividing pixel values by 255.
2. **Data Augmentation**: For the training set, various augmentation techniques are applied to enhance model generalization. These include:
   - Rotation
   - Width and height shifts
   - Zooming
   - Horizontal flipping
   - Brightness adjustments

3. **Class Balancing**: Class weights are calculated to address any imbalance in the dataset, ensuring that the model learns effectively from both classes.

## Relevant Statistics
- **Total Training Samples**: Approximately 5,863 images
- **Total Validation Samples**: Approximately 1,200 images
- **Total Test Samples**: Approximately 1,300 images
- **Class Distribution**:
  - Pneumonia: 3,583 images (training), 390 images (validation), 390 images (test)
  - Normal: 2,883 images (training), 410 images (validation), 410 images (test)

## Usage
The dataset is loaded and preprocessed using the `load_and_preprocess_data` function defined in the `src/dataloader/data_utils.py` file. This function handles the loading of images, applies necessary preprocessing, and prepares data generators for training, validation, and testing.

## Conclusion
This dataset documentation serves as a guide for understanding the dataset's structure, preprocessing, and usage within the Federated Learning project. Proper handling of the dataset is crucial for achieving optimal model performance and ensuring the validity of experimental results.