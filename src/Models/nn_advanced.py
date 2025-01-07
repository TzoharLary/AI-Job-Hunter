import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score


class AdvancedNNModel:
    def __init__(self, max_words=2000, max_seq_len=100, num_classes=5, lr=0.001):
        """
        max_words - size of the vocabulary (number of unique words) (tokenizer)
        max_seq_len - maximum length of a sequence (number of words in a sequence)
        num_classes - number of classes to predict
        """
        self.max_words = max_words
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.lr = lr
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_words, output_dim=64, input_length=self.max_seq_len))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(self.lr),
            metrics=['accuracy']
        )
        return model

    def train(self, X_train_pad, y_train, validation_split=0.1, epochs=5, batch_size=32):
        history = self.model.fit(
            X_train_pad, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history

    def predict(self, X):
        probs = self.model.predict(X)
        preds = np.argmax(probs, axis=1)
        return preds

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        return acc, prec, rec
