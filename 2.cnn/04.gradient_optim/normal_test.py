import torch
import torch.nn as nn

"""
BatchNormal LayerNormal InstanceNormal GroupNormal的使用

nn.BatchNorm2d(4) :选择一个通道，所有批次在这一个通道上进行计算，然后每个通道都这样计算一次
"""


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(2, 4, 3, 2, 1),

        )
        self.bn_layer = nn.Sequential(
            nn.BatchNorm2d(4)  # 选择一个通道，所有批次在这一个通道上进行计算，然后每个通道都这样计算一次
        )
        self.in_layer = nn.Sequential(
            nn.InstanceNorm2d(4)  # 在一个批次中，一个通道上对HW做计算
        )
        self.ln_layer = nn.Sequential(
            nn.LayerNorm([1, 4, 2, 2])  # 在一个批次中，对当前批次中的CHW进行计算，就是对当前图片做
            # 因为是在当前图片上做，所以需要知道图片的形状
        )
        self.gn_layer = nn.Sequential(
            nn.GroupNorm(num_groups=2, num_channels=4) # 在一个批次中，首先把通道分组，然后把每个组中按照
            # gHW进行单独进行归一化计算，最后把另一个组归一化计算后的数据合并到通道C上称为CHW
        )

    def forward(self, x):
        conv_out = self.conv_layer(x)  # [1 4 2 2]
        print(conv_out)
        bn_out = self.bn_layer(conv_out)
        print(bn_out.shape)
        in_out = self.in_layer(conv_out)
        print(in_out.shape)
        ln_out = self.ln_layer(conv_out)
        print(ln_out.shape)
        gn_out = self.gn_layer(conv_out)
        print(gn_out.shape)
        return in_out


if __name__ == '__main__':
    net = Net()
    # seed = torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 4)
    y = net(x)
    print(y.shape)
