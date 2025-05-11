# Models README

## Overview

This directory contains the implementations of various model architectures used in the federated learning experiments. The models are designed to handle image classification tasks, specifically targeting medical imaging datasets such as chest X-rays.

## Model Architectures

### Convolutional Neural Network (CNN)

- **File:** `cnn_model.py`
- **Description:** This file defines a convolutional neural network (CNN) architecture tailored for image classification. The model includes multiple convolutional layers, batch normalization, and dropout layers to enhance performance and prevent overfitting. The architecture is optimized for training on medical imaging datasets, providing a robust framework for feature extraction and classification.

## Intended Use Cases

The models implemented in this directory are intended for use in federated learning scenarios, where multiple clients train their models on local data and share updates with a central server. The CNN model is particularly suited for tasks involving medical image classification, such as detecting pneumonia in chest X-rays.

## Integration with Federated Learning

The models can be easily integrated into the federated learning framework defined in the `src/fed.py` script. Users can specify the model architecture to be used during the training process, allowing for flexibility in experimentation with different model configurations.

## Future Work

Future enhancements may include the implementation of additional model architectures, such as more complex deep learning models or transfer learning approaches, to further improve classification performance on diverse datasets.