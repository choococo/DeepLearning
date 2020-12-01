import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, f1_score, r2_score, precision_score

"""
nn.BCEWithLogitsLoss():自带sigmoid(自动对网络的输出进行sigmoid激活)
    适用于二分类中的内容
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.fc_layer(x)


if __name__ == '__main__':
    BATCH_SIZE = 100
    index = 34

    train_dataset = datasets.CIFAR10("./data", train=True, transform=img_transform, download=True)
    test_dataset = datasets.CIFAR10("./data", train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    net = Net().to(DEVICE)

    '输出[N,1]需要加上sigmoid进行激活'
    loss_func = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    count = 0
    for epoch in range(2):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = net(x)
            loss = loss_func(out, y)
            idx = torch.argmax(out, dim=1)
            accuracy = (torch.sum(torch.eq(idx, y)) / BATCH_SIZE).item()
            # print("损失为：", loss.item(), " | 训练集的当前精度为：", accuracy)
            # print("损失为：{:3f} | 训练集的当前精度为：{:4f}".format(loss.item(), accuracy))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


