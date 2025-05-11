# Results Summary for Federated Learning Project

This document summarizes the results obtained from the experiments conducted in the Federated Learning project. The project aims to explore various aggregation methods in a federated learning setting, particularly focusing on their performance in handling non-IID data distributions.

## Experiment Overview

The experiments were designed to evaluate the effectiveness of different aggregation methods under various non-IID factors. The following aggregation methods were implemented:

- **FedAvg**: Federated Averaging, which averages the weights from multiple clients.
- **FedMedian**: Federated Median, which calculates the median of the weights from multiple clients.
- **FedProx**: Federated Proximal, which adds a proximal term to the aggregation process to improve convergence.
- **FedTrimmedMean**: Federated Trimmed Mean, which removes a fraction of the largest and smallest weights before averaging.
- **FedNova**: Federated Normalized Averaging, which normalizes client updates to address statistical heterogeneity.
- **FedHeurAgg**: A novel aggregation method that dynamically adapts the importance of each client's contribution based on various factors.

## Performance Metrics

The following metrics were used to evaluate the performance of the models:

- **Test Accuracy**: The accuracy of the model on the test dataset.
- **Training Loss**: The loss value during training.
- **Validation Loss**: The loss value during validation.
- **Training Accuracy**: The accuracy of the model during training.
- **Validation Accuracy**: The accuracy of the model during validation.
- **Round Times**: The time taken for each communication round.

## Results Visualization

Visualizations were generated to provide insights into the performance of different aggregation methods across various non-IID factors. The following types of visualizations were included:

1. **Test Accuracy Comparison**: A line plot comparing the test accuracy of different methods across non-IID factors.
2. **Bar Chart Comparison**: A bar chart illustrating the test accuracy of each method for different non-IID factors.
3. **Learning Curves**: Plots showing the training and validation accuracy and loss over the epochs for each method and non-IID factor.
4. **Results Summary Table**: A summary table displaying the test accuracy and average round time for each method and non-IID factor.

## Key Findings

- The performance of aggregation methods varied significantly based on the non-IID factor.
- FedHeurAgg showed promising results in adapting to the data distribution and improving model performance.
- Visualizations provided a clear comparison of the effectiveness of each method, highlighting the strengths and weaknesses in different scenarios.

## Conclusion

The experiments conducted in this project demonstrate the potential of various aggregation methods in federated learning. The results indicate that adaptive methods like FedHeurAgg can enhance model performance, particularly in non-IID settings. Future work may explore further optimizations and additional aggregation strategies to improve federated learning outcomes.

## Results Directory

All results, including metrics and visualizations, are saved in the `results` directory. The following files are included:

- `accuracies.json`: A JSON file containing the simplified results with test accuracies.
- `metrics_<aggregation_method>_noniid<factor>.csv`: CSV files containing detailed metrics for each aggregation method and non-IID factor.
- `test_accuracy_comparison.png`: A PNG file of the test accuracy comparison plot.
- `test_accuracy_bar_chart.png`: A PNG file of the test accuracy bar chart.
- `learning_curves_<method>_noniid<factor>.png`: PNG files of the learning curves for each method and non-IID factor.
- `results_table.png`: A PNG file of the results summary table.

For further details on the implementation and usage, please refer to the main README file located in the root directory of the project.