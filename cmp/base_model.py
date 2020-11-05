import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class network_resnet50(nn.Module):   
    def __init__(self, num_classes = 10000):
        super(network_resnet50, self).__init__()
        self.resnet = resnet50(pretrained=False)
        self.new_resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=None)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=None)

    def forward(self, x):
        x = self.new_resnet(x)
    
        x1 = torch.squeeze(self.maxpool(x))
        x2 = torch.squeeze(self.avgpool(x))
        x = torch.cat((x1, x2), -1)

        return x


def resnet_50(**kwargs):
    model = network_resnet50(**kwargs)
    return model
