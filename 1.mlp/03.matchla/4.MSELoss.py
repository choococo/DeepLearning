import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, f1_score, r2_score, precision_score

"""
nn.MSELoss():
    要保证网络输出的形状out与标签的label的形状一致
"""

DEVICE = torch.device("cpu")

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.fc_layer(x)


if __name__ == '__main__':
    BATCH_SIZE = 100
    index = 34

    train_dataset = datasets.CIFAR10(r"D:\Dataset\cifar10", train=True, transform=img_transform, download=True)
    test_dataset = datasets.CIFAR10(r"D:\Dataset\cifar10", train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    net = Net().to(DEVICE)

    '输出[N,1]自动进行sigmoid激活'
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    avg_acc = []
    for i, (test_x, test_y) in enumerate(test_loader):
        test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)

        test_out = net(test_x)
        test_y = torch.eye(10)[test_y]
        loss = loss_func(test_out, test_y)

        print(loss)
