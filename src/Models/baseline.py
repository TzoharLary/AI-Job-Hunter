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

    def train(self, train_labels: list):
        """
        Train the baseline model by identifying the majority class.
        """
        train_counts = torch.tensor(train_labels).bincount()
        self.majority_class = train_counts.argmax().item()
        print(f"the majority class is {self.Category_mapping[self.majority_class]}")

    def predict(self, test_labels: list):

        # doesn't matter what the input is, always predict the majority class
        return np.full(shape=(len(test_labels),), fill_value=self.majority_class)


