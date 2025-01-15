import numpy as np
from data_preprocessing import DataPreprocessor
from metrics import calculate_accuracy, calculate_precision, calculate_recall
import torch

from Models.softmax import SoftmaxModel
from Models.baseline import BaselineModel
# from Models.nn_simple import SimpleNNModel
# from Models.nn_advanced import AdvancedNNModel

# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

def  main():
    # explanation of stage 1 - preprocessing:
    """
    1. we create an instance of the DataPreprocessor class that called preprocessor.
    2. we load the data, that is mean we read the data from the csv file and store it in the preprocessor object.
    3. we split the data into training and testing sets, this is done by calling the split_data method that she is in the DataPreprocessor class.
    """

    # 1) Preprocessing: Load data, define label,
    """"
        1. Create an instance of the DataPreprocessor class named preprocessor.
        2. Load the data from the CSV file.
        3. Define the label column as "title job".
        4. Convert the data to numbers
        5. convert the data to tensor
        
    """
    preprocessor = DataPreprocessor(csv_path="../data/Job Prediction By Resume.csv")

    preprocessor.load_data()

    data = preprocessor.dataset.data

    preprocessor.define_label("title job")

    Category_mapping = preprocessor.Tags_mapping    # Convert the data to numbers
    preprocessor.convert_csv_values(data)

    # preprocessor.visualize_data()
    # TODO: remove Tags field from datapreprocessor because he dont really pointer to y field in the dataset object

    train_dataset, val_dataset, test_dataset = preprocessor.dataset.get_datasets()
    train_Label = [y for _, y in train_dataset]
    train_Example = [x for x, _ in train_dataset]
    val_Label = [y for _, y in val_dataset]
    val_Example = [x for x, _ in val_dataset]
    test_Label = [y for _, y in test_dataset]
    test_Example = [x for x, _ in test_dataset]

    # this print really return the first 3 examples and labels of the train, validation, and test datasets
    """
    print(f"first 10 Examples and Labels: {train_Example[:3]} {train_Label[:3]}")
    print(f"first 10 Examples and Labels: {test_Example[:3]} {test_Label[:3]}")
    print(f"first 10 Examples and Labels: {val_Example[:3]} {val_Label[:3]}")
    """

    # Baseline model code:
    """
    baseline_model = BaselineModel(Category_mapping)
    baseline_model.train(train_Label)
    baseline_pred = baseline_model.predict(test_Example)

    print("=== Baseline Model ===")
    print(f"recall: {calculate_recall(test_Label, baseline_pred)}")
    print(f"precision: {calculate_precision(test_Label, baseline_pred)}")
    print(f"accuracy: {calculate_accuracy(test_Label, baseline_pred)}")
    """
    # print("the current type of data", type(data))
    # print("the current type of train_dataset", type(train_dataset))
    # print all the first 3 rows in data
    pri
    print(data.head())

    # input_dim, NumOfTags = preprocessor.dataset.get_num_of_XY()
    # print(f"input_dim: {input_dim}, NumOfTags: {NumOfTags}")
    # Softmax_model = SoftmaxModel(input_dim=input_dim, NumOfTags=NumOfTags, lr=0.01)

    # Softmax_model.train_model(preprocessor.dataset, num_epochs=1)

    # 3) Softmax model code:
    """
    input_dim, NumOfTags = data.get_num_of_XY()
    print(f"input_dim: {input_dim}, NumOfTags: {NumOfTags}")
    Softmax_model = SoftmaxModel(input_dim=input_dim, NumOfTags=NumOfTags, lr=0.01)
    # Train the model
    print("Training the SoftmaxModel...")
    Softmax_model.train_model(train_dataset, num_epochs=1)
    # Predict
    # print("Evaluating the SoftmaxModel...")
    # Softmax_pred = Softmax_model.predict(test_dataset)
    # accuracy = (Softmax_pred == test_Label).sum().item() / len(test_labels)
    # print(f"Accuracy: {accuracy:.4f}")
    """



"""
  

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
"""
if __name__ == "__main__":
    main()