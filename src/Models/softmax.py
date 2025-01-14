# Option of code that generate by the code generator of Github copilot
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class SoftmaxModel(nn.Module):
    def __init__(self, input_dim, num_classes, lr=0.01.__init__(), device="cpu"):
        """
        input_dim: number of features in the dataset.
        num_classes: num of classes in the output.
        lr: learning rate for the optimizer.
        """
        super(SoftmaxModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.device = torch.device(device)

        # Move the model to the device
        self.to(self.device)

    def forward(self, x):
        print("Forward pass input:", x.shape)
        output = self.linear(x)
        print("Forward pass output:", output.shape)
        return output

    def train_model(self, train_dataset, num_epochs=1):

        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            total_loss = 0

            for idx in range(len(train_dataset)):
                batch_x, batch_y = train_dataset[idx]
                batch_x = np.array(batch_x, dtype=np.float32)
                batch_x = torch.tensor(batch_x).to(self.device)
                batch_y = torch.tensor(np.array(batch_y), dtype=torch.long).to(self.device)
                print(f"Batch {idx + 1}: Input: {batch_x.shape}, Target: {batch_y.shape}")
                self.optimizer.zero_grad()
                logits = self.forward(batch_x.unsqueeze(0))
                loss = self.criterion(logits, batch_y.unsqueeze(0))

                print(f"Batch {idx + 1}, Loss: {loss.item():.4f}")

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_dataset):.4f}")

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



