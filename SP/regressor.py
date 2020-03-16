import torch
import torch.nn as nn

# 二分类网络，输出两张人脸是否是同一个人 / 是否属于同一家族
class Regressor(nn.Module):
    def __init__(self, n):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(n, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x
