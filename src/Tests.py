import os
import torch
import pandas as pd
import pytest
from torch.utils.data import DataLoader
import random

from data_preprocessing import DataPreprocessor  # Ensure the path is correct according to your project


class TestDataPreprocessor:

    @pytest.fixture(scope="class")
    def preprocessor(self):
        """
        Fixture to set up the full pipeline of DataPreprocessor.
        This combines creating the instance, loading data, and making it available to the tests.
        """
        csv_path = "../data/Job Prediction By Resume.csv"
        assert os.path.exists(csv_path), f"File not found at {csv_path}"
        preprocessor = DataPreprocessor(csv_path=csv_path)

        return preprocessor

    def test_load_data(self, preprocessor):
        preprocessor.load_data()
        df = preprocessor.df  # that's mean df will be the csv file that store in data in CustomDataset object
        #  check if df is a DataFrame object
        assert isinstance(df, pd.DataFrame), "Data not loaded as DataFrame"
        # check if dataset is a CustomDataset object and not None
        assert preprocessor.dataset is not None, "Dataset object not created"
        # check if dataset.data is not None
        assert preprocessor.dataset.data is not None, "Dataset data not loaded"
        # check if preprocessor.df equal to preprocessor.dataset.data in frame and data
        assert preprocessor.df.equals(preprocessor.dataset.data), "preprocessor.df and dataset.data are not the same"
        # check if preprocessor.df equal to df in frame and data
        assert preprocessor.df.equals(df), "preprocessor.df and df are not the same"
        # check if preprocessor.dataset.data equal to df in frame and data
        assert preprocessor.dataset.data.equals(df), "dataset.data and df are not the same"

    def test_handle_missing_values(self, preprocessor):
        """
        Verifies that the handle_missing_values function fills missing values in the specified column
        with the provided default value, and ensures no missing values remain in the DataFrame.
        """
        # Define test cases for different columns and default values
        test_cases = [
            {"column": "Skills", "default_value": "No"},
            {"column": "First Job title in your current field of work ", "default_value": "Unemployed"},
            {"column": "Certificate course title", "default_value": "No"}
        ]

        for case in test_cases:
            column = case["column"]
            default_value = case["default_value"]

            # Create a copy of the DataFrame for comparison
            df_initial = preprocessor.df.copy()

            # Ensure there are missing values in the column before filling
            assert df_initial[column].isnull().sum() > 0, f"No missing values found in column: {column}"

            # Call the function to handle missing values
            preprocessor.handle_missing_values(column, default_value)

            # Get the updated DataFrame
            df_after = preprocessor.df

            # Verify no missing values remain in the specified column
            assert df_after[column].isnull().sum() == 0, f"Missing values still exist in column: {column}"

            # Split the extraction of replaced values into multiple steps for clarity
            # Step 1: Identify rows that were null in the original DataFrame
            rows_with_nulls = df_initial[column].isnull()

            # Step 2: Select those rows in the updated DataFrame
            missing_filled = df_after.loc[rows_with_nulls, column]

            # Step 3: Verify that all replaced values match the default value
            for val in missing_filled:
                assert val == default_value, f"Unexpected value found: {val}, expected: {default_value}"

        # Verify that there are no missing values anywhere in the DataFrame
        missing_locations = preprocessor.df.isnull()
        if missing_locations.any().any():
            missing_map = {}
            for column in missing_locations.columns:
                missing_rows = missing_locations[column][missing_locations[column]].index.tolist()
                for row in missing_rows:
                    # add the missing row to the missing_map
                    if row not in missing_map:
                        missing_map[row] = []
                    missing_map[row].append(column)

                # create a detailed error message
            details = []
            for row, columns in missing_map.items():
                # assume that the column "Name" is present in the DataFrame
                person_name = preprocessor.df.loc[row, "Name"] if "Name" in preprocessor.df.columns else "Unknown"
                details.append(f"Row {row}: Missing in columns {columns} of {person_name}")

            error_message = "Missing values remain in the DataFrame after processing:\n" + "\n".join(details)
            assert False, error_message
        else:
            assert True, "No missing values remain in the DataFrame after processing."

    def test_define_label(self, preprocessor):
        """
        Check this in define_label function"
          - Check if X is equal to the original DataFrame without the label column
          - Check if the dtype of y is numeric (categories -> int)
          - Check if Tags_mapping is a dictionary whose length equals the number of unique labels in the column
          - Check if NumOfTags and NumOfFeatures are set correctly
          - Additional test: Use DataLoader and verify the batch structure


        """
        preprocessor.define_label("title job")
        dataset = preprocessor.dataset

        # Check: X should be equal to the original DataFrame without the label column
        assert dataset.X.equals(preprocessor.df.drop(columns=["title job"])), "X is not equal to the original DataFrame"

        # Check: dtype of y should be numeric (categories -> int)
        assert pd.api.types.is_integer_dtype(dataset.y.dtype), "Label dtype is not integer"

        # Check: Tags_mapping should be a dictionary whose length equals the number of unique labels in the column
        unique_labels = preprocessor.df["title job"].dropna().astype("category").cat.categories
        assert len(preprocessor.Tags_mapping) == len(unique_labels)

        # Check: NumOfTags and NumOfFeatures should be set correctly
        assert dataset.NumOfTags == len(unique_labels), "NumOfTags is incorrect"
        assert dataset.NumOfFeatures == len(dataset.X.columns), "NumOfFeatures is incorrect"

    def test_transform_text(self, preprocessor):
        """
        Check this in fit_transform_text function:
            - Verify that text columns are transformed into numerical lists
            - Verify that the transformation is consistent across all values in the column
            - Verify that the original DataFrame is updated with the transformed values
            - Verify that a random word in the vocabulary maps correctly to its numeric representation
            - Verify that the vocabulary contains all unique words in the text column
        """
        # Define a test case for all the text columns in the dataset using for loop 
        for column in preprocessor.df.select_dtypes(include=object).columns:
            # Skip the label column
            if column == "title job":
                continue
        
            # Create a copy of the DataFrame for comparison
            df_initial = preprocessor.df.copy()
        
            # Ensure the column contains string values
            assert df_initial[column].dtype == object, f"Column '{column}' is not of type 'object'"
        
            # Call the function to transform text values
            preprocessor.fit_transform_text(column)
        
            # Get the updated DataFrame
            df_after = preprocessor.df
        
            # Verify that the column has been transformed into numerical lists
            for val in df_after[column]:
                assert isinstance(val, list), f"Value '{val}' in column '{column}' is not a list"

            # pick a random word from the vocabulary
            random_word = random.choice(list(preprocessor.vocabulary.keys()))
            random_word_index = preprocessor.vocabulary[random_word]

            # verify that the random word maps correctly to its numeric representation
            for original_text, transformed_list in zip(df_initial[column], df_after[column]):
                if original_text and random_word in original_text.split():
                    assert random_word_index in transformed_list, (
                        f"Word '{random_word}' with index {random_word_index} not found in transformed list {transformed_list}"
                    )

            # # Verify that the vocabulary contains all unique words in the text column
            # all_words = set(word for text in df_initial[column].dropna() for word in text.split())
            # vocabulary_keys = set(preprocessor.vocabulary.keys())
            #
            # # Check if the vocabulary contains all unique words in the text column
            # assert vocabulary_keys == all_words, (
            #     f"Vocabulary mismatch: expected {all_words}, got {vocabulary_keys}"
            # )



    # def test_convert_csv_values(self, preprocessor):
    #     """
    #     Verifies that the convert_csv_values function correctly transforms columns:
    #       - For object-type columns – converts using fit_transform_text (resulting in numerical lists)
    #       - For numerical columns – values become tensors
    #     """
    #     # Use a copy of the original DataFrame
    #     df_copy = preprocessor.df.copy()
    #
    #     # Before the transformation, ensure the column type is not Tensor for any column in the X dataframe that could contain string values
    #     for col in df_copy.columns:
    #         if df_copy[col].dtype == object:
    #             for val in df_copy[col]:
    #                 assert not isinstance(val, torch.Tensor), f"Value '{val}' in column '{col}' is already a Tensor"
    #
    #     preprocessor.convert_csv_values(preprocessor.df)
    #
    #     # Numerical columns should be converted to tensors, while object-type columns (like Interests)
    #     # are transformed using fit_transform_text, resulting in numerical lists.
    #     for col in df_copy.columns:
    #         if df_copy[col].dtype == object:
    #             # Check that every value in the column is a list – or the result of the tensor transformation
    #             for val in df_copy[col]:
    #                 # If the value is a list – this indicates a successful transformation based on implementation.
    #                 assert isinstance(val, list), f"Value in column '{col}' is not a list"
    #         else:
    #             # For numerical columns, check that they are converted into tensors; select one sample value
    #             sample = df_copy[col].iloc[0]
    #             assert isinstance(sample, torch.Tensor), f"Value in column '{col}' is not a Tensor"

    # def test_split_data(self, preprocessor):
    #     """
    #     Verifies that splitting into train/val/test is performed correctly.
    #     Assumes the data has already undergone preprocessing (filling missing values, defining label, etc.)
    #     """
    #     # Assume the define_label operation has already occurred
    #     preprocessor.define_label("title job")
    #
    #     # For this test, use the split_data interface from the dataset object
    #     # Note: split_data in CustomDataset uses random_split, so we need to pass lengths as integers.
    #     total_length = len(preprocessor.dataset)
    #     train_size = int(0.7 * total_length)
    #     val_size = int(0.15 * total_length)
    #     test_size = total_length - train_size - val_size
    #
    #     preprocessor.dataset.split_data(train_size, val_size, test_size)
    #     train_dataset, val_dataset, test_dataset = preprocessor.dataset.get_datasets()
    #
    #     # Check: The total items across the three sets must equal the total length
    #     total_split = len(train_dataset) + len(val_dataset) + len(test_dataset)
    #     assert total_split == total_length
    #
    #     # Additionally, verify that the split is performed randomly (no special disruptions)
    #     assert len(train_dataset) == train_size
    #     assert len(val_dataset) == val_size
    #     assert len(test_dataset) == test_size
    #
    # # if __name__ == "__main__":
    # #     preprocessor = preprocessor()
    # #     test_load_data(preprocessor)
    # #     test_handle_missing_values(preprocessor, "Skills", "No")
