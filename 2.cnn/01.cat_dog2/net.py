import torch
import torch.nn as nn


class CatDogNet(nn.Module):

    def __init__(self):
        super(CatDogNet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(512 * 13 * 13, 2)
        )

    def forward(self, x):
        out = self.conv_layer(x)
        out = out.reshape(x.size(0), -1)
        out = self.fc_layer(out)
        return out


if __name__ == '__main__':
    x = torch.randn(2, 3, 100, 100)
    net = CatDogNet()
    y = net(x)
    print(y.shape)
