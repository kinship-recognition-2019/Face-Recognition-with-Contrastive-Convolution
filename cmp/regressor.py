import torch
import torch.nn as nn


class Regressor(nn.Module):
    def __init__(self, n):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(n, 100)
        self.fc2 = nn.Linear(100, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = nn.functional.dropout(x, p=0.01, training=self.training)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x