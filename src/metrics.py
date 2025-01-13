# This file will be used to contain performance measurement functions 
# such as accuracy, precision, recall, F1-score, and related visualizations.


# Explanation of metrics.py:
"""
PURPOSE:
This file contains functions to calculate and log various classification metrics such as
accuracy, precision, recall, and F1-score. It also contains utilities to compare the performance
of multiple models and visualize the results using bar charts.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import summary

# Function to calculate accuracy
def calculate_accuracy(true_labels, predictions, writer=None, step=0):
    """
    Calculate accuracy for the given true labels and predictions.
    
    Accuracy is defined as the ratio of correct predictions to the total number of samples.
    
    Log the metric to TensorBoard if a writer is provided.
    """
    accuracy = np.sum(np.array(true_labels) == np.array(predictions)) / len(true_labels)
    if writer:
        with writer.as_default():
            tf.summary.scalar("accuracy", accuracy, step=step)
    return accuracy * 100
# Function to calculate precision
def calculate_precision(true_labels, predictions, writer=None, step=0):
    """
    Calculate precision for the given true labels and predictions.
    
    Precision is the ratio of true positive predictions to the total number of positive predictions.
    This metric evaluates how many of the predicted positives are actually correct.
    
    Log the metric to TensorBoard if a writer is provided.
    """
    unique_labels, counts = np.unique(true_labels, return_counts=True)
    weights = counts / len(true_labels)
    precisions = []
    for label in unique_labels:
        tp = np.sum((np.array(predictions) == label) & (np.array(true_labels) == label))
        fp = np.sum((np.array(predictions) == label) & (np.array(true_labels) != label))
        precisions.append(tp / (tp + fp) if tp + fp > 0 else 0)
    precision = np.sum(np.array(precisions) * weights)
    if writer:
        with writer.as_default():
            tf.summary.scalar("precision", precision, step=step)
    return precision * 100

# Function to calculate recall - this recall weight by the relative frequency of each tag in the truth tags
def calculate_recall(true_labels, predictions, writer=None, step=0):
    """
    Calculate recall for the given true labels and predictions.
    
    Recall, also known as sensitivity or true positive rate, is the ratio of true positive predictions
    to the total number of actual positives in the dataset. This metric evaluates how well the model
    identifies true positives among all actual positives.
    
    Log the metric to TensorBoard if a writer is provided.
    """
    unique_labels, counts = np.unique(true_labels, return_counts=True)
    weights = counts / len(true_labels)
    recalls = []
    for label in unique_labels:
        tp = np.sum((np.array(predictions) == label) & (np.array(true_labels) == label))
        fn = np.sum((np.array(predictions) != label) & (np.array(true_labels) == label))
        recalls.append(tp / (tp + fn) if tp + fn > 0 else 0)
    recall = np.sum(np.array(recalls) * weights)
    if writer:
        with writer.as_default():
            tf.summary.scalar("recall", recall, step=step)
    return recall * 100

# Function to calculate F1-score
def calculate_f1_score(true_labels, predictions, writer=None, step=0):
    """
    Calculate F1-score for the given true labels and predictions.
    
    The F1-score is the harmonic mean of precision and recall. It provides a balance between
    the two metrics and is particularly useful when dealing with imbalanced datasets.
    
    Log the metric to TensorBoard if a writer is provided.
    """
    unique_labels, counts = np.unique(true_labels, return_counts=True)
    weights = counts / len(true_labels)
    f1_scores = []
    for label in unique_labels:
        tp = np.sum((np.array(predictions) == label) & (np.array(true_labels) == label))
        fp = np.sum((np.array(predictions) == label) & (np.array(true_labels) != label))
        fn = np.sum((np.array(predictions) != label) & (np.array(true_labels) == label))
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_scores.append(2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0)
    f1_score = np.sum(np.array(f1_scores) * weights)
    if writer:
        with writer.as_default():
            tf.summary.scalar("f1_score", f1_score, step=step)
    return f1_score

# Function to compare performance of multiple models
def compare_models_performance(metrics_dict, writer=None, step=0):
    """
    Compare performance for multiple models using their metrics.
    
    The function takes a dictionary of model names and their corresponding metrics (e.g., accuracy, 
    precision, recall, F1-score, etc.). It visualizes the performance metrics of all models as bar plots 
    for easy comparison. Optionally, the metrics are logged to TensorBoard.
    
    :param metrics_dict: A dictionary where keys are model names and values are dictionaries of metrics
                         (e.g., {"Model_1": {"accuracy": 0.9, "precision": 0.85, ...}, ...})
    """
    metrics = list(next(iter(metrics_dict.values())).keys())  # Get metric names
    models = metrics_dict.keys()
    
    # Prepare data for visualization
    data = {metric: [metrics_dict[model][metric] for model in models] for metric in metrics}
    
    if writer:
        with writer.as_default():
            for metric in data:
                for i, model in enumerate(models):
                    tf.summary.scalar(f"{model}/{metric}", data[metric][i], step=step)
    
    # Visualize each metric
    x = np.arange(len(models))  # Model indices
    width = 0.2  # Bar width
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        plt.bar(
            x + i * width, 
            data[metric], 
            width, 
            label=metric, 
            color=colors[i % len(colors)]
        )

    # Formatting the chart
    plt.xticks(x + width * (len(metrics) - 1) / 2, models)
    plt.xlabel('Models')
    plt.ylabel('Performance Metrics')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show or save the plot
    plt.tight_layout()
    plt.show()

