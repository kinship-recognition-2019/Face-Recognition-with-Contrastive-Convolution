import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        inner_channel = 4 * growth_rate

        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),

            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

class TailBottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        inner_channel = 4 * growth_rate

        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),

            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        print("in_channels:",in_channels)
        print("inner_channel:",inner_channel)
        return torch.cat([x, self.bottle_neck(x)], 1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False,stride=1),
            nn.AvgPool2d(3, stride=0,padding=1)
        )

    def forward(self, x):
        return self.down_sample(x)


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5):
        super().__init__()
        self.growth_rate = growth_rate


        inner_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False) 

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels
        #print("nblocks[len(nblocks)-1]",nblocks[len(nblocks)-1])
        self.dense_block = self._make_dense_layers(block,inner_channels,nblocks[len(nblocks)-1])
        self.tailBlock = TailBottleneck(1048,self.growth_rate)
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.bn = nn.BatchNorm2d(inner_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.dense_block(output)
        #print("output",output.size())
        #output = self.tailBlock(output)
        output = self.bn(output)
        output = self.relu(output)
        #print("output",output.size())
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

def densenet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def main():

    model = densenet121()  # .to(device)
    #print(model)
    x = torch.rand(1, 3, 128, 128)
    y = model(x)
    print(y.size())  # torch.Size([1, 512, 4, 4])


if __name__ == '__main__':
    main()