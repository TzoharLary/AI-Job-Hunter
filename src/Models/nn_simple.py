import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score


class SimpleNNModel:
    def __init__(self, input_dim, num_classes, lr=0.001):
        """
        input_dim = number of features in the input data
        num_classes = number of classes to predict
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(self.input_dim,)))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(self.lr),
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, validation_split=0.1, epochs=10, batch_size=32):
        history = self.model.fit(
            X_train, y_train,
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
