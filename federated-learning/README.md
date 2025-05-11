# Federated Learning Aggregation for Chest X RAYS

## Overview
This research implements a federated learning framework designed for image classification tasks, specifically using chest X-ray images to detect pneumonia. The framework includes various aggregation methods, data loading utilities, custom metrics, and model architectures. The goal is to enable collaborative learning across multiple clients while preserving data privacy.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Aggregation Methods](#aggregation-methods)
- [Metrics](#metrics)
- [Results](#results)
- [License](#license)

## Installation
To set up the project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd federated-learning
pip install -r requirements.txt
```

## Usage
To run the federated learning experiments, execute the main script:

```bash
python src/fed.py
```

You can modify the configuration settings in `src/config.py` to adjust hyperparameters, dataset paths, and other settings.

## Data
The dataset used in this project is the Chest X-ray dataset, which can be found at [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). The data is organized into training, validation, and test sets. Preprocessing steps include data augmentation and normalization.

## Models
The project includes a convolutional neural network (CNN) architecture designed for binary classification tasks. The model is defined in `src/models/cnn_model.py` and can be customized as needed.

## Aggregation Methods
The following aggregation methods are implemented in the project:

- **FedAvg**: Federated Averaging method that averages the weights from multiple clients.
- **FedMedian**: Federated Median method that calculates the median of the weights from multiple clients.
- **FedProx**: Federated Proximal method that adds a proximal term to improve convergence.
- **FedTrimmedMean**: Federated Trimmed Mean method that removes a fraction of the largest and smallest weights before averaging.
- **FedNova**: Federated Normalized Averaging method that normalizes client updates to address statistical heterogeneity.
- **FedHeurAgg**: A novel aggregation method that dynamically adapts the importance of each client's contribution based on various factors.

## Metrics
Custom metrics, including a custom F1 score, are implemented to evaluate model performance. These metrics can be found in `src/metrics/f1_score.py`.

## Results
Results from the experiments, including performance metrics and visualizations, are saved in the `results` directory. You can find detailed summaries and plots in the respective `results/README.md` file.

