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

    # def __getitem__(self, index):
    #     row = self.data.iloc[index]
    #     features = row["numbers"]
    #     label = row["label"]
    #     return features, label

    def __getitem__(self, index):
        if self.y is not None:
            return self.X.iloc[index].values, self.y.iloc[index]
        return self.X.iloc[index].values

    def get_num_of_XY(self):
        return self.NumOfFeatures, self.NumOfTags

    @staticmethod
    def custom_collate_fn(batch):
        """
        Function to pad lists of varying lengths and convert to tensors.
        Expects a batch to be a list of tuples in the form (features, label).
        """
        # Find the maximum length among all features in the batch
        max_length = max(len(item[0]) for item in batch)

        # Pad all features to the same length
        padded_numbers = [
            list(item[0]) + [0] * (max_length - len(item[0])) for item in batch
        ]

        # Convert the padded numbers to a tensor
        numbers_tensor = torch.tensor(padded_numbers, dtype=torch.float32)

        # Convert the labels to a tensor
        labels_tensor = torch.tensor([item[1] for item in batch], dtype=torch.int64)

        return numbers_tensor, labels_tensor

    def get_dataloaders(self, batch_size):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.custom_collate_fn)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.custom_collate_fn)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.custom_collate_fn)
        return train_loader, val_loader, test_loader

    def get_dataloader(self, dataset_name, batch_size):
        if dataset_name == "train":
            return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.custom_collate_fn)
        elif dataset_name == "val":
            return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.custom_collate_fn)
        elif dataset_name == "test":
            return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.custom_collate_fn)

    def get_datasets(self):
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_dataset(self, dataset_name):
        if dataset_name == "train":
            return self.train_dataset
        elif dataset_name == "val":
            return self.val_dataset
        elif dataset_name == "test":
            return self.test_dataset
