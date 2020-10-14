import torch.nn.functional as F
import torch
import torch.nn as nn
class Triple_loss(nn.Module):
    def __init__(self,alpha):
        super().__init__()
        self.alpha=alpha

    def forward(self, pos_dist,neg_dist):
        basic_loss = pos_dist-neg_dist+self.alpha
        loss = torch.mean(F.relu(basic_loss), 0)
        return loss
