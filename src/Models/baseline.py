import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


class BaselineModel:
    """
    baseline model is a model that always predicts the majority class
    """

    def __init__(self):
        self.majority_class = None

    def train(self, y_train):
        # find the majority class
        values, counts = np.unique(y_train, return_counts=True)
        idx = np.argmax(counts)
        self.majority_class = values[idx]

    def predict(self, X):
        # doesnt matter what the input is, always predict the majority class
        return np.full(shape=(len(X),), fill_value=self.majority_class)

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        return acc, prec, rec
