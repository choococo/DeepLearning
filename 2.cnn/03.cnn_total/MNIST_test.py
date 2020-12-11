import torch.nn as nn
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),  # N,16,28,28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # N,16,14,14

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, groups=8),  # N,64,14,14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2))  # N,64,7,7

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, groups=16),  # N,128,7,7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2))  # N,128,3,3

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=10, kernel_size=3, stride=1)  # N,10,1,1

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        return y


if __name__ == '__main__':
    net = Net()
    summary(net.to("cuda"), input_size=(1, 28, 28))

