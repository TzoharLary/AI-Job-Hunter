import numpy as np
from data_preprocessing import DataPreprocessor
from Models.baseline import BaselineModel
from Models.logistic_regression import LogisticRegressionModel
from Models.nn_simple import SimpleNNModel
from Models.nn_advanced import AdvancedNNModel

from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def main():
    # explanation of stage 1 - preprocessing:
    """
    1. we create an instance of the DataPreprocessor class that called preprocessor.
    2. we load the data, that is mean we read the data from the csv file and store it in the preprocessor object.
    3. we split the data into training and testing sets, this is done by calling the split_data method that she is in the DataPreprocessor class.
    """

    # 1) Preprocessing - load, define label, and split data.
    preprocessor = DataPreprocessor(csv_path="Job Prediction By Resume.csv")
    preprocessor.load_data()
    preprocessor.define_label("Job Title")
    train_df, validation_df, test_df = preprocessor.split_data()

    # explanation of stage 2 - training and testing the models:
    """
    1. we make 2 variables:
        X_train_text = the text data from the training set.
        X_test_text = the text data from the testing set.
    2. we make 2 variables:
        y_train = the labels from the training set.
        y_test = the labels from the testing set.
    """
    # Separate features and labels
    X_train_text = train_df["combined_text"]
    X_test_text = test_df["combined_text"]

    y_train = train_df["job_label"].values
    y_test = test_df["job_label"].values

    # 2) Baseline:
    """
    1. we create an instance of the BaselineModel class that called
    2. we train the baseline model using the train method that returns the majority class.
    
    """
    baseline_model = BaselineModel()
    baseline_model.train(y_train)
    y_pred_baseline = baseline_model.predict(X_test_text)
    baseline_acc, baseline_prec, baseline_rec = baseline_model.evaluate(y_test, y_pred_baseline)
    print("=== Baseline Model ===")
    print(f"Accuracy: {baseline_acc:.3f}, Precision: {baseline_prec:.3f}, Recall: {baseline_rec:.3f}\n")

    # 3) Logistic Regression
    # Vectorization
    X_train_tfidf = preprocessor.fit_transform_text(X_train_text)
    X_test_tfidf = preprocessor.transform_text(X_test_text)

    lr_model = LogisticRegressionModel()
    lr_model.train(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)
    lr_acc, lr_prec, lr_rec = lr_model.evaluate(y_test, y_pred_lr)
    print("=== Logistic Regression ===")
    print(f"Accuracy: {lr_acc:.3f}, Precision: {lr_prec:.3f}, Recall: {lr_rec:.3f}\n")

    # 4) Simple NN
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    num_classes = len(np.unique(y_train))
    simple_nn_model = SimpleNNModel(input_dim=X_train_dense.shape[1], num_classes=num_classes)
    simple_nn_model.train(X_train_dense, y_train, epochs=5)  # You can adjust epochs
    y_pred_nn = simple_nn_model.predict(X_test_dense)
    nn_acc, nn_prec, nn_rec = simple_nn_model.evaluate(y_test, y_pred_nn)
    print("=== Simple Neural Network ===")
    print(f"Accuracy: {nn_acc:.3f}, Precision: {nn_prec:.3f}, Recall: {nn_rec:.3f}\n")

    # 5) Advanced RNN
    # Assume we use Tokenizer to convert text to sequences
    tokenizer = Tokenizer(num_words=2000, oov_token="<UNK>")
    tokenizer.fit_on_texts(X_train_text)
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    max_seq_len = 100
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_len, padding='post', truncating='post')

    rnn_model = AdvancedNNModel(max_words=2000, max_seq_len=max_seq_len, num_classes=num_classes)
    rnn_model.train(X_train_pad, y_train, epochs=3)  # You can adjust epochs
    y_pred_rnn = rnn_model.predict(X_test_pad)
    rnn_acc, rnn_prec, rnn_rec = rnn_model.evaluate(y_test, y_pred_rnn)
    print("=== Advanced RNN Model ===")
    print(f"Accuracy: {rnn_acc:.3f}, Precision: {rnn_prec:.3f}, Recall: {rnn_rec:.3f}\n")

    print("Completed all models. Finished!")

if __name__ == "__main__":
    main()
