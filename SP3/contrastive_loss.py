import torch.nn.functional as F
import torch
import torch.nn as nn

class Contrastive_loss(nn.Module):
    def __init__(self,margin):
        super().__init__()
        self.margin=margin

    def forward(self, dist,y_true):
        # 类内损失：
        within_loss =F.relu(self.margin-dist)**2*y_true
        # 类间损失：
        between_loss = (1 - y_true)* dist
        # 总体损失（要最小化）：
        loss = 0.5 * torch.mean(within_loss + between_loss)
        return loss

