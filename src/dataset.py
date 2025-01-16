import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np


# Dataset implementation
class CustomDataset(Dataset):
    def __init__(self, csv_path):

        self.csv_path = csv_path
        self.data = pd.read_csv(self.csv_path)
        self.X = None
        self.y = None
        self.batch_size = None
        self.NumOfFeatures = None
        self.NumOfTags = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.X is not None and self.y is not None:
            return self.X.iloc[idx].values, self.y.iloc[idx]
        elif self.X is not None:
            return self.X.iloc[idx].values
        else:
            raise ValueError("Dataset is not properly initialized. Ensure X and y are assigned.")

    def get_dataloader(self, dataset, batch_size, shuffle=True):
        """
        Create batches of data for training or testing.
        """
        data = dataset.values
        if shuffle:
            np.random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            X_batch = [item[:-1] for item in batch]
            y_batch = [item[-1] for item in batch]
            yield np.array(X_batch), np.array(y_batch)

