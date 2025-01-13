import numpy as np
from src.dataset import CustomDataset, DatasetManager
from src.data_preprocessing import DataPreprocessor

from collections import Counter


class BaselineModel:
    """
    baseline model is a model that always predicts the majority class
    """

    def __init__(self, data: DataPreprocessor):
        self.majority_class = None
        self.data = data


    def train(self):
        """
        1. values  = the unique values in the Tags column
        2. counts = the number of times each value appears in the Tags column
        """
        # find the majority class
        values, counts = np.unique(self.data.Tags, return_counts=True)
        idx = np.argmax(counts)
        # majority class is the number that represents the location in the label list
        self.majority_class = values[idx]
        # print the name of the majority class
        # (use this line: self.category_mapping = dict(enumerate(self.dataset.data[label_col].astype("category").cat.categories)) that in DataPreprocessor class
        print(f"the majority class is {self.data.category_mapping[self.majority_class]}")

    def predict(self, X):
        # doesnt matter what the input is, always predict the majority class
        return np.full(shape=(len(X),), fill_value=self.majority_class)

    # def evaluate(self, y_true, y_pred):
    #     acc = accuracy_score(y_true, y_pred)
    #     prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    #     rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    #     return acc, prec, rec
