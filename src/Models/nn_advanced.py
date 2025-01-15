import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from src.data_preprocessing import DataPreprocessor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.metrics import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score
class SimpleNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01,device = None):
        # Initialize the SimpleNNModel class as a subclass of nn.Module
        super(SimpleNNModel, self).__init__()
        # Calls the parent class's __init__ method to initialize the model

        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        # Define the first fully connected (linear) layer (input_dim → hidden_dim)

        self.relu1 = nn.ReLU()
        # Define the ReLU activation function for non-linearity

        self.output = nn.Linear(hidden_dim, output_dim)
        # Define the output fully connected layer (hidden_dim → output_dim)

        self.criterion = nn.CrossEntropyLoss()
        # Define the loss function for multi-class classification
        # TODO check if better to use ADAM or SGD as we did in softmax
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # Define the optimizer (Adam) for updating model parameters with a learning rate

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Create a device object to specify where calculations will be performed (CPU or GPU)

        self.to(self.device)
        # Move model parameters (weights, biases) to the specified device

    def forward(self, x):
        x = torch.relu(self.hidden1(x)) # Apply hidden layer + ReLU
        x = self.output(x)   # Apply output layer
        return x     # Return raw logits

    # def _build_model(self):
    #     model = Sequential()
    #     model.add(Dense(256, activation='relu', input_shape=(self.input_dim,)))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(128, activation='relu'))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(self.num_classes, activation='softmax'))
    #     model.compile(
    #         loss='sparse_categorical_crossentropy',
    #         optimizer=tf.keras.optimizers.Adam(self.lr),
    #         metrics=['accuracy']
    #     )
    #     return model

    def train_model(self, train_loader, val_loader, num_epochs=10):
        """
        Train the model using the training data loader and validate it on the validation data loader.

        Args:
            train_loader: DataLoader object for the training data.
            val_loader: DataLoader object for the validation data.
            num_epochs: Number of epochs to train the model.

        Returns:
            None
        """
        # Set the model to training mode
        self.train()
        criterion = self.criterion  # Loss function already defined in the model
        optimizer = self.optimizer  # Optimizer already defined in the model

        for epoch in range(num_epochs):
            total_loss = 0.0

            # Iterate over training data
            for inputs, labels in train_loader:
                # Move data to the appropriate device
                inputs = inputs.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                # Zero the gradients to prevent accumulation
                optimizer.zero_grad()

                # Forward pass: compute predictions and loss
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Accumulate loss for logging
                total_loss += loss.item()

            # Print epoch progress
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

    def predict(self, x):
        x = self.forward(x)  # Get raw logits
        probabilities = F.softmax(x, dim=1)  # Apply Softmax to logits
        return probabilities

    # def evaluate(self, y_true, y_pred):
    #     acc = accuracy_score(y_true, y_pred)
    #     prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    #     rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    #     return acc, prec, rec

def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test data.
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total  # Return accuracy percentage


if __name__ == "__main__":

    # Load and preprocess the data
    preprocessor = DataPreprocessor(csv_path="../../data/Job Prediction By Resume.csv")
    preprocessor.load_data()
    preprocessor.define_label("title job")  # Replace with your label column name
    dataset = preprocessor.dataset
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=32)

    # Define input and output dimensions
    input_dim, output_dim = dataset.get_num_of_XY()
    hidden_dim = 64
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the model
    model = SimpleNNModel(input_dim, hidden_dim, output_dim, lr, device)

    # Start training
    print("Starting training...")
    model.train_model(train_loader, val_loader, num_epochs=10)
    print("Training complete.")
    # # Evaluate the model on test data
    # print("Evaluating the model on test data...")
    # test_accuracy = evaluate_model(model, test_loader)
    #
    # # Get true labels and predictions for additional metrics
    # true_labels, predicted_labels = [], []
    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = model(inputs)
    #         _, preds = torch.max(outputs, 1)
    #         true_labels.extend(labels.cpu().numpy())
    #         predicted_labels.extend(preds.cpu().numpy())
    #
    # # Calculate additional metrics
    # accuracy = calculate_accuracy(true_labels, predicted_labels)
    # precision = calculate_precision(true_labels, predicted_labels)
    # recall = calculate_recall(true_labels, predicted_labels)
    # f1_score = calculate_f1_score(true_labels, predicted_labels)
    #
    # print(f"Accuracy: {accuracy:.2f}%")
    # print(f"Precision: {precision:.2f}%")
    # print(f"Recall: {recall:.2f}%")
    # print(f"F1-Score: {f1_score:.2f}")
