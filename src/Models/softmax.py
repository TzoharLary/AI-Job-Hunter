# Option of code that generate by the code generator of Github copilot

import torch
import torch.nn as nn
import torch.optim as optim


class SoftmaxModel(nn.Module):
    def __init__(self, input_dim, num_classes, lr=0.01):
        """
        input_dim: number of features in the dataset.
        num_classes: num of classes in the output.
        lr: learning rate for the optimizer.
        """
        super(SoftmaxModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        print("Forward pass input:", x.shape)
        output = self.linear(x)
        print("Forward pass output:", output.shape)
        return output

    def train_model(self, train_dataset, num_epochs=10):
        train_loader = datasetManager.get_dataloader("train", batch_size=100)
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            total_loss = 0
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                print(f"Batch {batch_idx + 1}: batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}")
                self.optimizer.zero_grad()
                logits = self.forward(batch_x)
                loss = self.criterion(logits, batch_y)
                print(f"Batch {batch_idx + 1}: Loss: {loss.item()}")
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



