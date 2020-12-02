import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, f1_score, r2_score, precision_score
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备

img_transform = transforms.Compose([  # 训练集的
    transforms.ToTensor(),  # 转成CHW的(0-1)的张量
    transforms.RandomCrop(32, padding=4),  # 对数据进行扩充，扩充4个像素，随机进行裁剪
    transforms.RandomHorizontalFlip(),  # 进行垂直随机0.5进行翻转
    # transforms.ColorJitter(contrast=[1.1, 1.5]),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 将数据归到[-1,1]之间
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
    index = 41
    dataset_path = r"D:\Dataset\cifar10"

    train_dataset = datasets.CIFAR10(dataset_path, train=True, transform=img_transform, download=True)
    test_dataset = datasets.CIFAR10(dataset_path, train=False, transform=test_transform, download=True)
    # for image, tag in train_dataset:
    #     img = transforms.ToPILImage()(image)
    #     print(type(img))
    #     # img.show()
    #     # exit(0)
    #     img = img_transform(img)
    #     img = transforms.ToPILImage()(img)
    #     img.show()
    #     exit(0)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    net = Net().to(DEVICE)

    if os.path.exists(f"params5/{index}.t"):
        net.load_state_dict(torch.load(f"params5/{index}.t"))




    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

    count = 0
    # count_list = []
    # accuracy_list = []
    # loss_list = []
    # plt.ion()
    # for epoch in range(3):
    #     for i, (x, y) in enumerate(train_loader):
    #         x, y = x.to(DEVICE), y.to(DEVICE)
    #
    #         out = net(x)
    #         loss = loss_func(out, y)
    #         idx = torch.argmax(out, dim=1)
    #         accuracy = (torch.sum(torch.eq(idx, y)) / BATCH_SIZE).item()
    #         # print("损失为：{:3f} | 训练集的当前精度为：{:4f}".format(loss.item(), accuracy))
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #     avg_acc = []
    #     net.eval()
    #     for i, (test_x, test_y) in enumerate(test_loader):
    #         test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
    #
    #         test_out = net(test_x)
    #
    #         loss = loss_func(test_out, test_y)
    #
    #         test_idx = torch.argmax(test_out, dim=1)
    #         accuracy = (torch.sum(torch.eq(test_idx, test_y)) / BATCH_SIZE).item()
    #         # print(accuracy_score(test_y.detach().cpu().numpy(), test_idx.detach().cpu().numpy()))
    #         avg_acc.append(accuracy)
    #         # 0.594199986755848
    #     torch.save(net.state_dict(), f"params5/{epoch + index + 1}.t")
    #     print("保存模型成功!")
    #     print("epoch:{}  | 平均精度为：{}".format(epoch, np.mean(avg_acc)))
    #     if np.mean(avg_acc) >= 0.609:
    #         torch.save(net.state_dict(), f"params5/{epoch + index + 1}_{count}.t")
    #
    #         count += 1
    #         count_list.append(count)
    #         accuracy_list.append(accuracy)
    #         loss_list.append(loss.item())
    #
    #         # 绘制图形
    #         plt.clf()
    #         plt.subplot(1, 2, 1)
    #         plt.scatter(count_list, loss_list, s=50)
    #         plt.subplot(1, 2, 2)
    #         plt.scatter(count_list, accuracy_list, s=50)
    #         plt.pause(0.01)
    #     torch.save(net.state_dict(), f"params3/{epoch + index + 1}.t")
    # plt.ioff()
    # plt.show()

    avg_acc = []
    net.eval()
    for i, (test_x, test_y) in enumerate(test_loader):
        test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)

        test_out = net(test_x)

        loss = loss_func(test_out, test_y)

        test_idx = torch.argmax(test_out, dim=1)
        accuracy = (torch.sum(torch.eq(test_idx, test_y)) / BATCH_SIZE).item()
        # print(accuracy_score(test_y.detach().cpu().numpy(), test_idx.detach().cpu().numpy()))

        avg_acc.append(accuracy)
    print("最终平均精度为：", np.mean(avg_acc))

    """
    训练了10个epoch：
    平均精度为： 22 : 0.594199986755848
                    0.5687999868392944
               30 : 0.5884999868273735
               35 : 0.5898999854922294
               36 : 0.6037999886274338
               41 ： 0.6083999896049499
    """
