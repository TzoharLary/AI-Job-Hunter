from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


class LogisticRegressionModel:
    def __init__(self, multi_class="multinomial", solver="saga", max_iter=1000):
        self.model = LogisticRegression(
            multi_class=multi_class,
            solver=solver,
            max_iter=max_iter
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        return acc, prec, rec
