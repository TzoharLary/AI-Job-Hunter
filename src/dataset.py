import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader


# Dataset implementation
class CustomDataset(Dataset):
    def __init__(self, data_path, dataset_name):
        """
        PURPOSE:
            Initialize the dataset with the data path and dataset name.

        HOW:
            1. Load the data from the given path.
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.data = pd.read_csv(self.data_path)
        self.X = None
        self.y = None
        self.batch_size = None
        self.NumOfFeatures = None
        self.NumOfTags = None

        train_size = int(0.7 * len(self))
        val_size = int(0.15 * len(self))
        test_size = len(self) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self, [train_size, val_size, test_size]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.y is not None:
            return self.X.iloc[index].values, self.y.iloc[index]
        return self.X.iloc[index].values

    def get_num_of_XY(self):
        return self.NumOfFeatures, self.NumOfTags

    def get_dataloaders(self, batch_size):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def get_dataloader(self, dataset_name, batch_size):
        if dataset_name == "train":
            return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        elif dataset_name == "val":
            return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        elif dataset_name == "test":
            return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def get_datasets(self):
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_dataset(self, dataset_name):
        if dataset_name == "train":
            return self.train_dataset
        elif dataset_name == "val":
            return self.val_dataset
        elif dataset_name == "test":
            return self.test_dataset
