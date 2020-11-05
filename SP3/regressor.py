import torch
import torch.nn as nn

# 二分类网络，输出两张人脸是否是同一个人 / 是否属于同一家族
class Regressor(nn.Module):
    def __init__(self, n):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(n, 1)
        self.relu = nn.ReLU()
        self.nan_to_zero = Nan_to_zero()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        #x = self.nan_to_zero(x)
        return x

class Nan_to_zero(nn.Module):
    def __init__(self):
        super(Nan_to_zero, self).__init__()

    def forward(self, x):
        zero = torch.full(x.size(),1e-30).to(torch.device("cuda"))
        out = torch.where(torch.isnan(x), zero, x)
        return out
