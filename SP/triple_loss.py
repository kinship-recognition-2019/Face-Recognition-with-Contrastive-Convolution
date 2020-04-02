import torch.nn.functional as F
import torch
import torch.nn as nn
class Triple_loss(nn.Module):
    def __init__(self,alpha):
        super().__init__()
        self.alpha=alpha

    def forward(self, pos_dist,neg_dist):

        #pos_dist = torch.sum(torch.pow((Ab_list - B_list), 2),1)
        #neg_dist = torch.sum(torch.pow((Ac_list - C_list), 2),1)
        relu=nn.ReLU(inplace=True)
        basic_loss = pos_dist-neg_dist+self.alpha
        #print("basic_loss",basic_loss.size())
        #print(basic_loss.size())
        loss = torch.mean(relu(basic_loss), 0)
        #print("loss",loss.size())
        return loss

