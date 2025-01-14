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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.y is not None:
            return self.X.iloc[index].values, self.y.iloc[index]
        return self.X.iloc[index].values

class DatasetManager:
    # Explanation of this class:
    """
            1. split the dataset into training, validation, and testing sets. (using the random_split method)
            2. Create 3 DataLoaders:
                a. train_loader: DataLoader for the training set.
                b. val_loader: DataLoader for the validation set.
                c. test_loader: DataLoader for the testing set.
                3. Do visualization of the data in this way:
                    STAGE 1: Collecting the data:
                            ACTION: Divide each dateset group into:
                                a. X - features
                                b. Y - labels
                            RESULT: train_data becomes (X_train, y_train) etc.
                    STAGE 2: Converting X and y to tensors and then to numpy arrays:
                            ACTION: Convert the X and y to tensors
                            CODE: X_train = torch.stack(X_train)
                            ACTION: Convert the tensors to numpy arrays
                            CODE: X_train = torch.stack(X_train).numpy()
                            RESULT: X_train, y_train etc. are now tensors
                    STAGE 3: Creating scatter plots:
                            ACTION: Create graphs for each dataset with uniform size and share axes
                            CODE: fig, (train_ax, validation_ax, test_ax) = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(15, 5))
                            RESULT: 3 scatter plots are created

                            ACTION 2: Draw the scatter plots in the created graphs in different ways and add titles and labels

                            ACTION 3: Show the scatter plots on the screen
                            CODE: plt.show()
    """

# 
    def __init__(self, df: CustomDataset):
        self.data = df
        train_size = int(0.7 * len(self.data))
        val_size = int(0.15 * len(self.data))
        test_size = len(self.data) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.data, [train_size, val_size, test_size]
        )

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

    def Convert_Features(self, features):

        # create Featured_Name to contain list of the first row in the X field
        Featured_Name = self.data.X[0]






