from dataset import CustomDataset


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
        self.dataset = None

# Explanation load_data method: (Purpose and how it works)
    """
    PURPOSE: Read the data from the CSV file, clean it.
    
    HOW:
    1. we read the data from the csv file and store it in the df variable named self.df. (using the read_csv method)

    """

    def load_data(self):
        """
        PURPOSE: Use CustomDataset to load and handle the dataset automatically.
        """
        self.dataset = CustomDataset(self.csv_path, "text_dataset")

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

