from collections import Counter
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

from src.dataset import CustomDataset
class DataPreprocessor:
    def __init__(self, csv_path):
        self.vocabulary = {}
        self.csv_path = csv_path
        self.dataset = None
        self.df = None
        self.Tags_mapping = {}
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.Tag_column = None

    def load_data(self):
        self.dataset = CustomDataset(self.csv_path, "text_dataset")
        self.df = self.dataset.data

    def clean_column_names(self):
        self.df.columns = (self.df.columns
                           .str.strip()
                           .str.replace(r"\s*\(.*?\)", "", regex=True)
                           .str.replace(r"[^\w\s]", "_", regex=True)
                           .str.replace(r"\s+", "_", regex=True)
                           .str.lower())

    def handle_missing_values(self, column_name, fill_value):
        if column_name not in self.df.columns:
            raise KeyError(f"Column '{column_name}' not found in dataset.")
        self.df[column_name] = self.df[column_name].fillna(fill_value)

    def handle_multivalue_cells(self):

        for column in self.df.select_dtypes(include=['object']).columns:
            self.df[column] = self.df[column].apply(
                lambda x: x.split(";") if pd.notnull(x) and ";" in str(x) else (str(x) if pd.notnull(x) else [])
            )

    def define_label(self, Tag_column):
        self.Tag_column = Tag_column
        self.dataset.data = self.df.dropna(subset=[Tag_column])
        self.dataset.y = self.df[Tag_column]
        self.dataset.X = self.df.drop(columns=[Tag_column, "match_percentage"])  # Dropping match_percentage column
        self.dataset.y = self.dataset.y.astype("category").cat.codes
        self.Tags_mapping = dict(
            enumerate(
                self.dataset.data[Tag_column].astype("category").cat.categories
            )
        )
        self.dataset.NumOfTags = len(self.Tags_mapping)
        self.dataset.NumOfFeatures = len(self.dataset.X.columns)

    def split_data(self, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
        np.random.seed(seed)
        indices = np.random.permutation(len(self.df))
        train_end = int(train_frac * len(self.df))
        val_end = train_end + int(val_frac * len(self.df))
        self.train_dataset = self.df.iloc[indices[:train_end]]
        self.val_dataset = self.df.iloc[indices[train_end:val_end]]
        self.test_dataset = self.df.iloc[indices[val_end:]]

    def get_datasets(self, name):

        if name == "train":
            return self.train_dataset
        elif name == "val":
            return self.val_dataset
        elif name == "test":
            return self.test_dataset
        else:
            raise ValueError("Dataset name is not valid")

    def get_test_Label(self):
        return  self.test_dataset[self.Tag_column]

    def tokenize_texts(self, texts, max_vocab_size=10000):
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        all_tokens = [token for text in tokenized_texts for token in text]
        vocab_counter = Counter(all_tokens).most_common(max_vocab_size)
        self.vocabulary = {word: idx + 1 for idx, (word, _) in enumerate(vocab_counter)}

        return [[self.vocabulary.get(word, 0) for word in text] for text in tokenized_texts]

    def pad_sequences(self, sequences):

        maxlen = max(len(seq) for seq in sequences)
        for i in range(len(sequences)):
            sequences[i] = sequences[i] + [0] * (maxlen - len(sequences[i]))
        return sequences

    def preprocess_dataset(self):

        for column in self.df.columns:
            if self.df[column].dtype == 'object' or self.df[column].apply(lambda x: isinstance(x, str)).any():
                texts = self.df[column].tolist()
                value_types = Counter(type(value).__name__ for value in texts)
                for value_type, count in value_types.items():
                    print(f"{value_type}: {count}")
                    return 0
                tokenized_texts = self.tokenize_texts(texts)
                padded_texts = self.pad_sequences(tokenized_texts)
                self.df[column] = padded_texts
