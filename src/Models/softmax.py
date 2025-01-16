# Option of code that generate by the code generator of Github copilot

import torch
import torch.nn as nn
import torch.optim as optim


class SoftmaxModel(nn.Module):
    def __init__(self, input_dim, NumOfTags, lr=0.01.__init__(), CPU="cpu"):
        """
        input_dim: number of features in the dataset.
        NumOfTags: num of classes in the output.
        lr: learning rate for the optimizer.
        """
        super(SoftmaxModel, self).__init__()
        # self.linear is a layer that contain this components:
        # Weight matrix:
        #               Shape: (num_classes, input_dim)
        #               Initialized randomly
        # Bias vector:
        #               Shape: (num_classes)
        #               Initialized randomly
        # This layer perform the operation: y = Wx + b where:
        #   x is the input vector
        #   W is the weight matrix
        #   b is the bias vector
        #   y is the output vector
        # input layer
        # self.linear = nn.Linear(input_dim, NumOfTags)

        input1 = 5
        input2 = 3
        self.linear = nn.Linear(input1, input2)
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # TODO: check if we need to replace SGD with the attitude of bathces
        # Optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

        # this row creates a "device" object that
        # defines where the calculations will be performed.
        self.device = torch.device(CPU)
        # Move the parameters of the model (weights, biases) to the device
        self.to(self.device)

    def forward(self, x):
        print("Forward pass input:", x.shape)
        output = self.linear(x)
        print("Forward pass output:", output.shape)
        return output

    def train_model(self, data, num_epochs=1):
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            total_loss = 0
            train_loader = data.get_dataloader("train", batch_size=100)
            # print the type of the batch
            print("Data type:", type(data))
            print("Batch type:", type(train_loader))


            for idx, (batch_x, batch_y) in enumerate(train_loader):

                # Move the batch to the device
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.long)
                # print(f"Batch {idx + 1}: Input: {batch_x.shape}, Target: {batch_y.shape}")

                # Zero the gradients for avoid situation of accumulating the gradients
                self.optimizer.zero_grad()

                # Forward pass

                logits = self.forward(batch_x)
                print("Logits:", logits.shape)

                # Compute negative log-likelihood loss manually
                loss = self.criterion(logits, batch_y)

                print(f"Batch {idx + 1}, Loss: {loss.item():.4f}")

                # Backward pass and optimization
                loss.backward()



                self.optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_loader):.4f}")

    def predict(self, x):
        with torch.no_grad():
            print("Prediction input:", x.shape)
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
            print("Predictions:", predictions)
            return predictions



"""
import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

    class SoftmaxModel(nn.Module):
        def __init__(self, vocab_size, num_features, num_classes):
            super(SoftmaxModel, self).__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, num_features, sparse=True)
            self.linear = nn.Linear(num_features, num_classes)
        
        def preprocess_text(self, text, vocab, tokenizer):
            return torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long)
        
        def forward(self, text, numbers):
            text_features = self.embedding(text)
            combined = torch.cat((text_features, numbers), dim=1)
            return self.linear(combined)
"""



