import torch
import torch.nn as nn


def extractpatches(x, patch_size):
    patches = x.unfold(2, patch_size,  1).unfold(3, patch_size, 1)
    bs, c, pi, pj, _, _ = patches.size()

    l = [patches[:, :, int(i / pi), i % pi, :, :] for i in range(pi * pi)]
    f = [l[i].contiguous().view(-1, c * patch_size * patch_size) for i in range(pi * pi)]

    stack_tensor = torch.stack(f)
    stack_tensor = stack_tensor.permute(1,0,2)
    return stack_tensor


class GenModel(nn.Module):
   def __init__(self,feature_size ):
       super(GenModel,self).__init__()
       self.f_size = feature_size
       self.g1 = nn.Linear(self.f_size*3*3, self.f_size*3*3)
       self.g2 = nn.Linear(self.f_size*2*2, self.f_size*3*3)
       self.g3 = nn.Linear(self.f_size*1*1, self.f_size*3*3)
       self.relu = nn.ReLU()
       self.conv3x3 = nn.Conv2d(self.f_size,self.f_size,3)

   def forward(self, x):
       bs, _, _, _= x.size()
       S0 = x
       p1 = extractpatches(S0,3)
       S1 = self.relu(self.conv3x3(S0))
       p2 = extractpatches(S1,2)
       S2 = self.relu(self.conv3x3(S1))
       p3 = extractpatches(S2,1)
       kk1 = self.relu(self.g1( p1))
       kk2 = self.relu(self.g2( p2))
       kk3 = self.relu(self.g3( p3))
       kernels1 = torch.cat((kk1, kk2,kk3), dim = 1)
       return kernels1
