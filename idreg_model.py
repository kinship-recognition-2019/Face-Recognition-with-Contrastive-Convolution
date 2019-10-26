import torch.nn as nn
import torch.nn.functional as F


class Identity_Regressor(nn.Module):
    def __init__(self, n, classes):
        super(Identity_Regressor, self).__init__()
        self.fc1 = nn.Linear(n, 256)
        self.fc2 = nn.Linear(256, classes)

    def forward(self, x):
        bs, m, n = x.size()
        x = x.view(-1,n*m)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
