import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader


# Dataset implementation
class CustomDataset(Dataset):
    def __init__(self, data_path, dataset_name):
        """
        PURPOSE:
            Initialize the dataset with the data path and dataset name.

        HOW:
            1. Load the data from the given path.
            2. Split the data into X (features) and y (labels).
        """

        self.data_path = data_path
        self.dataset_name = dataset_name
        self.data = pd.read_csv(self.data_path)
        self.X = None
        self.y = None

        # Automatically handle splitting if labels exist
        if "label" in self.data.columns:
            self.X = self.data.drop(columns=["label"])
            self.y = self.data["label"]
        else:
            self.X = self.data

    def __len__(self):
        return len(self.X)

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

    def __init__(self, data_path):

        self.data = CustomDataset(data_path, "full_dataset")
        train_size = int(0.7 * len(self.data))
        val_size = int(0.15 * len(self.data))
        test_size = len(self.data) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.data, [train_size, val_size, test_size]
        )

    def get_dataloaders(self, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def visualize_data(self, train_data, validation_data, test_data):
        """
        PURPOSE:
            Visualize the training, validation, and testing datasets using scatter plots.

        HOW:
            1. Extract features and labels from the datasets.
            2. Use matplotlib to create scatter plots for each dataset.
            3. Label the axes and add titles for clarity.
        """
        # STAGE 1: Collecting the data: split Type_data into X_Type and y_Type
        X_train, y_train = zip(*[(x, y) for x, y in train_data])
        X_validation, y_validation = zip(*[(x, y) for x, y in validation_data])
        X_test, y_test = zip(*[(x, y) for x, y in test_data])

        # Stage 2: Converting X and y to tensors and then to numpy arrays:
        X_train = torch.stack(X_train).numpy()
        y_train = torch.tensor(y_train).numpy()

        X_validation = torch.stack(X_validation).numpy()
        y_validation = torch.tensor(y_validation).numpy()

        X_test = torch.stack(X_test).numpy()
        y_test = torch.tensor(y_test).numpy()

        # Stage 3: Creating scatter plots:
        fig, (train_ax, validation_ax, test_ax) = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(15, 5))
        train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral)
        train_ax.set_title("Training Data")
        train_ax.set_xlabel("Feature #0")
        train_ax.set_ylabel("Feature #1")

        validation_ax.scatter(X_validation[:, 0], X_validation[:, 1], c=y_validation)
        validation_ax.set_title("Validation Data")
        validation_ax.set_xlabel("Feature #0")

        test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        test_ax.set_title("Testing Data")
        test_ax.set_xlabel("Feature #0")

        plt.show()
