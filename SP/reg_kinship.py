import torch
import torch.nn as nn

# 二分类网络，输出两张人脸是否是同一个人 / 是否属于同一家族
class RegressorKinship(nn.Module):
    def __init__(self, n):
        super(RegressorKinship, self).__init__()
        self.fc1 = nn.Linear(n, 2560)
        self.fc2 = nn.Linear(2560, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
