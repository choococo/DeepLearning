import torch
import torch.nn as nn


class Convolutional(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Convolutional, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.sub_module(x)


class DownSampling(nn.Module):

    def __init__(self, in_channels):
        super(DownSampling, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, 2, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.sub_module(x)


class NetTempV1(nn.Module):

    def __init__(self):
        super(NetTempV1, self).__init__()
        self.conv_layer = nn.Sequential(
            Convolutional(3, 8, 3, 1, 1),
            DownSampling(8),
            Convolutional(16, 16, 1, 1, 0),
            Convolutional(16, 32, 3, 1, 1),
            DownSampling(32),
            Convolutional(64, 64, 1, 1, 0),
            Convolutional(64, 128, 3, 1, 1),
            DownSampling(128),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 6 * 6, 256),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        out = self.conv_layer(x)
        out = out.reshape(x.size(0), -1)
        out = self.fc_layer(out)
        return out


if __name__ == '__main__':
    net = NetTempV1()
    x = torch.randn(2, 3, 100, 100)
    y = net(x)
    print(y.shape)
