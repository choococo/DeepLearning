import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score


class Net(nn.Module):  # 定义网络模型

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 256),  # 线性层
            nn.BatchNorm1d(256),  # 归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(256, 128),  # 线性层
            nn.BatchNorm1d(128),  # 归一化
            nn.ReLU(inplace=True)  # 激活函数
        )
        self.out = nn.Linear(128, 10)  # 输出层

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # 形状变换(N,C,H,W)->[N,V]
        out = self.fc(x)
        out = self.out(out)
        return out


def visualize(loss_list, acc_list, count):
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.title("Accuracy")
    plt.xlabel("epoch")
    plt.ylim(0, 1)
    # plt.ylabel("%")
    # plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['20', '40', '60', '80', '100'])
    plt.scatter(count, acc_list, c='orange', s=50)
    # plt.plot(count, acc_list, c='orange')

    plt.subplot(1, 2, 2)
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(0, 2.7)
    plt.scatter(count, loss_list, s=50)
    # plt.plot(count, loss_list)
    plt.pause(0.01)


def validation(net, test_loader, device, loss_func, BATCH_SIZE):
    net.eval()  # 进入测试模式
    loss_list = []
    acc_list = []
    for i, (x, y) in enumerate(test_loader):  # 加载测试集
        x, y = x.to(device), y.to(device)  # 放入到cuda上
        out = net(x)  # 网络输出
        loss = loss_func(out, y)  # 损失
        idx = torch.argmax(out, dim=1)
        accuracy = (torch.sum(torch.eq(idx, y)) / BATCH_SIZE).item()
        acc_list.append(accuracy)
        loss_list.append(loss.item())
        if i % 100:
            print("accuracy : ", accuracy)
    print("平均精度为： ", np.mean(acc_list))
    print("平均损失为： ", np.mean(loss_list))
    print(len(acc_list))


def train(net, train_loader, device, loss_func, optimizer, save_params, BATCH_SIZE):
    net.train()  # 模型训练
    j = 0
    acc_list = []
    count = []
    loss_list = []
    before_loss = 0
    plt.ion()  # 打开实时画图
    for epoch in range(2):  # 开始训练
        for i, (x, y) in enumerate(train_loader):  # 加载训练集
            x, y = x.to(device), y.to(device)  # 将数据放到cuda上
            out = net(x)  # 获取网络输出 [100,10]

            loss = loss_func(out, y)  # 损失计算
            optimizer.zero_grad()  # 梯度清空
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度更新
            idx = torch.argmax(out, dim=1)
            accuracy = (torch.sum(torch.eq(idx, y)) / BATCH_SIZE).item()
            # accuracy = torch.mean(idx == y) # mean通常用与对多个精度或损失求平均值
            y = y.detach().cpu().numpy()
            idx = idx.detach().cpu().numpy()
            # print(accuracy_score(y, idx))
            # print(precision_score(y, idx, average="macro"))
            # print(precision_score(y, idx, average="micro"))
            # print(precision_score(y, idx, average="weighted"))
            # exit(0)
            if before_loss < loss:
                torch.save(net.state_dict(), save_params)
            else:
                before_loss = loss
            j += 1
            acc_list.append(accuracy)
            count.append(j)
            loss_list.append(loss.item())

            visualize(loss_list, acc_list, count)
            # print(f1_score(y, idx, average="micro"))
    plt.ioff()  # 关闭实时画图
    plt.show()  # 显示图片


if __name__ == '__main__':
    BATCH_SIZE = 100
    if not os.path.exists("save_params"):  # 创建文件夹
        os.mkdir("save_params")
    save_params = r"./save_params/params.pth"
    save_net = r"./save_params/net.pth"

    transform = transforms.Compose([  # 数据预处理
        transforms.ToTensor(),  # [H,W,C]->[C,H,W]
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    # 数据集下载
    train_dataset = datasets.MNIST("../data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST("../data", train=False, transform=transform, download=False)
    # 数据集加载
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 当前设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # 把网络模型放到GPU上
    net = Net().to(device)

    loss_func = nn.CrossEntropyLoss()  # 损失函数设计
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)  # 优化器设置

    # 判断是否存在权重文件
    if os.path.exists(save_params):
        net.load_state_dict(torch.load(save_params))
        print("参数加载成功！")
    else:
        print("No Params!")
    print(len(train_loader))
    print(len(test_loader))

    isTrain = True

    if isTrain:
        train(net, train_loader, device, loss_func, optimizer, save_params, BATCH_SIZE)
    else:
        validation(net, test_loader, device, loss_func, BATCH_SIZE)

"""
shift + tab 向前缩紧
Total params: 235,914
Trainable params: 235,914
Non-trainable params: 0
"""
