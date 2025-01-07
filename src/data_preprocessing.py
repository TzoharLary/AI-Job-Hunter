import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader


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
        self.csv_path = csv_path
        self.df = None
        self.label_col = None

# Explanation load_data method: (Purpose and how it works)
    """
    PURPOSE: Read the data from the CSV file, clean it.
    
    HOW:
    1. we read the data from the csv file and store it in the df variable named self.df. (using the read_csv method)

    """
    def load_data(self):
        # Read the CSV
        self.df = pd.read_csv(self.csv_path)

        ## IM not sure if i want to make new column with combined text of Interests, Skills, etc.
        # # Create a new column with combined text (Interests, Skills, etc.) if desired
        # self.df["combined_text"] = self.df.apply(self._combine_text_features, axis=1)

# Explanation define_label method: (Purpose and how it works)
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
    def define_label(self, label_col):
        # Define the label column
        self.label_col = label_col
        # Handle null values in the label column
        self.df = self.df.dropna(subset=[self.label_col])
        # Convert the label to numbers
        self.df["job_label"] = self.df[self.label_col].astype('category').cat.codes

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
    def _combine_text_features(self, row):
        text_parts = []
        columns_for_text = ["Interests", "Skills", "Certificate course title",
                            "Work in the past", "First Job title in your current field of work"]
        for col in columns_for_text:
            if col in self.df.columns and isinstance(row[col], str):
                text_parts.append(row[col])
        return " ".join(text_parts)

# Explanation split_data method: (Purpose and how it works)
    """
    PURPOSE: Split the data into training, validation, and testing sets.
    
    HOW: 
    1. we define the sizes for the training, validation, and testing sets.
    2. we use the random_split method to split the data into the training, validation, and testing sets.
       NOTE: we use the random_state parameter to ensure that the split is reproducible.
    3. we return the training, validation, and testing sets.
    """
    def split_data(self, random_state=42):
        # Define sizes
        train_size = int(0.7 * len(self.df))
        validation_size = int(0.15 * len(self.df))
        test_size = int(0.15 * len(self.df))
        # Split the data and keep consistently.
        train_df, validation_df, test_df = random_split(
            self.df,
            [train_size, validation_size, test_size],
            generator=torch.Generator().manual_seed(random_state)
        )
        return train_df, validation_df, test_df

# Explanation visualize_data method: (Purpose and how it works)

    def visualize_data(self, train_data, validation_data, test_data):
        """
        PURPOSE:
            Visualize the training, validation, and testing datasets using scatter plots.

        HOW:
            1. Extract features and labels from the datasets.
            2. Use matplotlib to create scatter plots for each dataset.
            3. Label the axes and add titles for clarity.
        """
        X_train, y_train = zip(*[(x, y) for x, y in train_data])
        X_validation, y_validation = zip(*[(x, y) for x, y in validation_data])
        X_test, y_test = zip(*[(x, y) for x, y in test_data])

        X_train = torch.stack(X_train).numpy()
        y_train = torch.tensor(y_train).numpy()

        X_validation = torch.stack(X_validation).numpy()
        y_validation = torch.tensor(y_validation).numpy()

        X_test = torch.stack(X_test).numpy()
        y_test = torch.tensor(y_test).numpy()

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



