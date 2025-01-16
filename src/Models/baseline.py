import numpy as np
from src.dataset import CustomDataset
from src.data_preprocessing import DataPreprocessor
import torch

from collections import Counter


class BaselineModel:
    """
    baseline model is a model that always predicts the majority class
    """

    def __init__(self, Category_mapping: dict):
        self.Category_mapping = Category_mapping
        self.majority_class = None

    def train(self, train_labels):
        train_counts = Counter(train_labels)
        inverse_mapping = {v: k for k, v in self.Category_mapping.items()}
        train_counts = Counter([inverse_mapping[label] for label in train_labels])
        self.majority_class = max(train_counts, key=train_counts.get)
        print(f"The majority class is {self.Category_mapping[self.majority_class]}")

    def predict(self, test_labels):

        return [self.majority_class] * len(test_labels)

