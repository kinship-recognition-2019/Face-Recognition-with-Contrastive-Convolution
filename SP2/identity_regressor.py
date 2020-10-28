import torch
import torch.nn as nn
import torch.nn.functional as F

# 多分类网络，输出单张人脸所属的人脸编号 / 家庭编号

class Identity_Regressor(nn.Module):
    def __init__(self, n, classes):
        super(Identity_Regressor, self).__init__()
        self.fc1 = nn.Linear(n, 256)
        self.fc3 = nn.Linear(256, classes)

    def forward(self, x):
        bs, m, n = x.size()
        x = x.view(-1, n*m)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x