import torch
import torch.nn as nn


class Regressor(nn.Module):
    def __init__(self, n):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(n, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x
