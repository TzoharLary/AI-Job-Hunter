from src.dataset import CustomDataset, DatasetManager
import torch
import numpy as np
import matplotlib.pyplot as plt

# Explanation DataPreprocessor class: (Purpose and methods)
"""
PURPOSE: Create an object that preprocesses the data while he's get the data from the csv file.

Contain the following methods:
1. load_data: Read the data from the CSV file, clean it, and convert the job titles to numerical values.
2. combine_text_features: Combine text features from different columns into a single text for each row.
   This method is used only if we want to create a new column with combined text.
3. split_data: Split the data into training, validation, and testing sets.
4. fit_transform_text: Fit and transform the text data using the TF-IDF vectorizer.
5. transform_text: Transform the text data using the TF-IDF vectorizer.

add thia method:
1. method that handle null values in the First Job title column.
"""
class DataPreprocessor:
    def __init__(self, csv_path):
        self.vocabulary = {}
        self.csv_path = csv_path
        self.dataset = None
        self.datasetManager = None
        self.Features = None
        self.Tags = None

    def load_data(self):
        """
        PURPOSE: Use CustomDataset to load and handle the dataset automatically.
        """
        # Load the data using create an instance of the CustomDataset class and store it in the dataset field.
        self.dataset = CustomDataset(self.csv_path, "text_dataset")
        self.Features = self.dataset.X
        self.Tags = self.dataset.y


    def define_label(self, label_col):
        """
           PURPOSE:
               1. Define the label column for the classification task
               2. Convert the labels to numbers.

           HOW:
               1. gets parameter that contains the name of the label column and store it in the label_col field.
               2. we remove rows where there is no job title (the label) using the dropna method.
               3. we convert the label (job title) to numbers using the fit_transform method.
               this is useful because machine learning models work with numbers, not text.
               for example, if we have a job title like 'data scientist', 'software engineer', etc., we can convert them to numbers like 0, 1, etc.

       """
        # Define the label column using the init of CustomDataset, that's mean define the field y of the dataset as the label column
        self.dataset.data = self.dataset.data.dropna(subset=[label_col])  # Apply dropna to the DataFrame
        """
        1. i define the field y of the dataset object as the value that is store in the 
        dataset = field in this class that we load into the CustomDataset object.
        dataset.y = if dataset is CustomDataset object, then dataset.y = the value of the field y in dataset.
        dataset.data = the csv file
        """
        # define the y field of the dataset object as the column that here columne name is like the label_col
        self.dataset.y = self.dataset.data[label_col]
        self.dataset.X = self.dataset.data.drop(columns=[label_col])  # Update the features

        # Convert the label to numbers
        self.dataset.y = self.dataset.y.astype("category").cat.codes
        # Store the mapping between the original labels and the numerical values
        self.category_mapping = dict(enumerate(self.dataset.data[label_col].astype("category").cat.categories))

        # print(f"the label is converted to numbers as  {self.dataset.y}")
        self.datasetManager = DatasetManager(self.dataset)
        train_dataset, val_dataset, test_dataset = self.datasetManager.get_datasets()

    def convert_to_tensors(self):

        # For loop about all the columns in X to fit_transform_text
        # for col in self.dataset.X.columns:
        #     self.fit_transform_text(col)


        # Convert the data to tensors
        self.dataset.Features = torch.tensor(self.dataset.Features.values, dtype=torch.float32)
        self.dataset.Tags = torch.tensor(self.dataset.Tags.values, dtype=torch.long)
        print("Data converted to tensors.")

    def _combine_text_features(self, row):
        # Explanation combine_text_features method: (Purpose and how it works)
        """
            PURPOSE: Combine text features from different columns into a single text for each row.

            HOW:
            1. we create an empty list called text_parts.
            2. we create a list of column names that we want to combine.
            3. we iterate over the columns in the list.
            4. if the column is in the dataframe and the value is a string, we add it to the text_parts list.
            5. we use the join method to combine the text parts into a single text.
            6. we return the combined text.
        """
        text_parts = []
        columns_for_text = ["Interests", "Skills", "Certificate course title",
                            "Work in the past", "First Job title in your current field of work"]
        for col in columns_for_text:
            if col in self.df.columns and isinstance(row[col], str):
                text_parts.append(row[col])
        return " ".join(text_parts)

    # add every word to vocabulary array for text column.
    def fit_transform_text(self, text_column):
        """
        PURPOSE:
            Tokenize and vectorize the text data using a custom vocabulary-based vectorizer.

        HOW:
            1. Tokenize the text data by splitting it into words.
            2. Build a vocabulary by assigning a unique index to each word.
            3. Convert each text entry into a numerical vector based on word frequencies.
        """
        df = self.dataset.data
        # all_text will contain only the rows with value
        all_text = df[text_column].dropna().tolist()
        for text in all_text:
            for word in text.split():
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)

    # TODO: Add statistic visualization like:
    # - Most 3 common labels
    def visualize_data(self):

        # Step 1: Split the dataset
        total_size = len(self.dataset)
        train_count = int(0.7 * total_size)
        val_count = int(0.15 * total_size)
        test_count = total_size - train_count - val_count

        train_dataset, val_dataset, test_dataset = self.datasetManager.get_datasets()

        # Step 2: Extract labels
        train_labels = [y for _, y in train_dataset]
        val_labels = [y for _, y in val_dataset]
        test_labels = [y for _, y in test_dataset]

        # Step 3: Count occurrences of each label
        train_counts = torch.tensor(train_labels).bincount()
        val_counts = torch.tensor(val_labels).bincount()
        test_counts = torch.tensor(test_labels).bincount()

        # Normalize counts to percentages
        train_dist = train_counts / train_counts.sum()
        val_dist = val_counts / val_counts.sum()
        test_dist = test_counts / test_counts.sum()


        # Step 4: Visualize distributions
        labels = list(range(len(train_counts)))
        label_names = [self.category_mapping[i] for i in labels]
        width = 0.25

        print("\nDetailed Label Distributions:")
        for i, label in enumerate(label_names):
            train_percentage = train_dist[i].item() * 100
            val_percentage = val_dist[i].item() * 100
            test_percentage = test_dist[i].item() * 100
            print(f"{label}: Train {train_percentage:.3f}%, Val {val_percentage:.3f}%, Test {test_percentage:.3f}%")

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