import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CatDogNet(nn.Module):

    def __init__(self):
        super(CatDogNet, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(30000, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.fc_layer(x)
        return self.out(out)  # [N, 2]


if __name__ == '__main__':
    x = torch.randn(2, 3, 100, 100)
    net = CatDogNet()
    y = net(x)
    print(y.shape)
