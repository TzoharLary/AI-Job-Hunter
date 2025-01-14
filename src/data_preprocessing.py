import matplotlib.pyplot as plt
import torch

from src.dataset import CustomDataset


class DataPreprocessor:
    def __init__(self, csv_path):
        """
        PURPOSE: Create an object that preprocesses the data while he's get the data from the csv file.
        """
        self.vocabulary = {}
        self.csv_path = csv_path
        self.dataset = None  # CustomDataset object





        self.Features = None
        self.Tags = None

        # Save category mapping so we know which job name corresponds to which numerical category.
        self.Tags_mapping = {}

    # TODO: remove the dataset_name parameter from the CustomDataset constructor because it's not used
    def load_data(self):
        """
        PURPOSE: Use CustomDataset to load and handle the dataset automatically.
        """
        self.dataset = CustomDataset(self.csv_path, "text_dataset")
        self.Features = self.dataset.X
        self.Tags = self.dataset.y

    def define_label(self, label_col):
        """
        PURPOSE:
           1. Define the label column for the classification task
           2. Convert the labels to numbers (categories).

        HOW:
           1. Receives the column name to be used as the label (label_col), and filters out rows without values (dropna).
           2. Sets self.dataset.y as the values of that column.
           3. Saves the rest in self.dataset.X
           4. Converts the column values to numerical categories and saves inverted mapping (Tags_mapping).
        """
        # Remove rows that have no value in label_col
        self.dataset.data = self.dataset.data.dropna(subset=[label_col])

        # Define label and X
        self.dataset.y = self.dataset.data[label_col]
        self.dataset.X = self.dataset.data.drop(columns=[label_col])

        # Convert to categorical (numeric) values
        self.dataset.y = self.dataset.y.astype("category").cat.codes

        # Save mapping (number -> original category name)
        self.Tags_mapping = dict(
            enumerate(
                self.dataset.data[label_col].astype("category").cat.categories
            )
        )
        self.dataset.NumOfTags = len(self.Tags_mapping)
        self.dataset.NumOfFeatures = len(self.dataset.X.columns)

        # Initialize local X and y
        self.Features = self.dataset.X
        self.Tags = self.dataset.y

    def convert_to_tensors(self):
        """
        PURPOSE: 
            Convert the loaded features and tags to PyTorch tensors.
        """
        # Convert to tensor
        self.dataset.Features = torch.tensor(
            self.dataset.X.values, dtype=torch.float32
        )
        self.dataset.Tags = torch.tensor(
            self.dataset.y.values, dtype=torch.long
        )
        print("Data converted to tensors.")

    def _combine_text_features(self, row):
        """
        PURPOSE: Combine text features from different columns into a single text for each row.
        """
        text_parts = []
        columns_for_text = [
            "Interests",
            "Skills",
            "Certificate course title",
            "Work in the past",
            "First Job title in your current field of work",
        ]
        for col in columns_for_text:
            if col in self.df.columns and isinstance(row[col], str):
                text_parts.append(row[col])
        return " ".join(text_parts)

    def fit_transform_text(self, text_column):
        """
        PURPOSE:
            Tokenize and vectorize the text data using a custom vocabulary-based vectorizer.
        """
        df = self.dataset.data
        all_text = df[text_column].dropna().tolist()

        for text in all_text:
            for word in text.split():
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)

    def visualize_data(self):
        """
        PURPOSE:
            Provides a basic visualization of label distribution
            across train, val, and test splits.
        """
        # Extract the three parts of the dataset from CustomDataset
        train_dataset, val_dataset, test_dataset = self.dataset.get_datasets()

        # Extract the labels for each part
        train_labels = [y for _, y in train_dataset]
        val_labels = [y for _, y in val_dataset]
        test_labels = [y for _, y in test_dataset]

        # Count values for each part
        train_counts = torch.tensor(train_labels).bincount()
        val_counts = torch.tensor(val_labels).bincount()
        test_counts = torch.tensor(test_labels).bincount()

        # Convert to percentages
        train_dist = train_counts / train_counts.sum()
        val_dist = val_counts / val_counts.sum()
        test_dist = test_counts / test_counts.sum()

        labels = list(range(len(train_counts)))
        label_names = [self.Tags_mapping[i] for i in labels]
        width = 0.25

        print("\nDetailed Label Distributions:")
        for i, label in enumerate(label_names):
            train_percentage = train_dist[i].item() * 100
            val_percentage = val_dist[i].item() * 100
            test_percentage = test_dist[i].item() * 100
            print(
                f"{label}: Train {train_percentage:.3f}%, "
                f"Val {val_percentage:.3f}%, "
                f"Test {test_percentage:.3f}%"
            )

        plt.bar([x - width for x in labels], train_dist, width=width, label="Train")
        plt.bar(labels, val_dist, width=width, label="Validation")
        plt.bar([x + width for x in labels], test_dist, width=width, label="Test")
        plt.xlabel("Labels")
        plt.ylabel("Percentage")
        plt.title("Label Distribution in Train, Validation, and Test Sets")
        plt.legend()
        plt.show()

    def transform_text(self, text_column):
        """
        PURPOSE:
            Transform new text data into numerical vectors using the pre-built vocabulary.
        """
        df = self.dataset.data

        def vectorize(text):
            vector = [0] * len(self.vocabulary)
            for word in text.split():
                if word in self.vocabulary:
                    vector[self.vocabulary[word]] += 1
            return vector

        df[text_column] = df[text_column].fillna("").apply(vectorize)
