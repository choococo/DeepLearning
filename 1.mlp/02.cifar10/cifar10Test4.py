import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, f1_score, r2_score, precision_score

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

    train_dataset = datasets.CIFAR10("./data", train=True, transform=img_transform, download=True)
    test_dataset = datasets.CIFAR10("./data", train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    net = Net().to(DEVICE)

    if os.path.exists(f"params4/{index}.t"):
        net.load_state_dict(torch.load(f"params4/{index}.t"))

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

    # count = 0
    # # count_list = []
    # # accuracy_list = []
    # # loss_list = []
    # # plt.ion()
    # for epoch in range(2):
    #     for i, (x, y) in enumerate(train_loader):
    #         x, y = x.to(DEVICE), y.to(DEVICE)
    #
    #         out = net(x)
    #         loss = loss_func(out, y)
    #         idx = torch.argmax(out, dim=1)
    #         accuracy = (torch.sum(torch.eq(idx, y)) / BATCH_SIZE).item()
    #         # print("损失为：", loss.item(), " | 训练集的当前精度为：", accuracy)
    #         # print("损失为：{:3f} | 训练集的当前精度为：{:4f}".format(loss.item(), accuracy))
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
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
    #     torch.save(net.state_dict(), f"params4/{epoch + index + 1}.t")
    #     print("保存模型成功!")
    #     print("epoch:{}  | 平均精度为：{}".format(epoch, np.mean(avg_acc)))
    #     if np.mean(avg_acc) >= 0.598:
    #         torch.save(net.state_dict(), f"params4/{epoch + index + 1}_{count}.t")
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
               26 : 0.593300010919571
               27 : 0.5966999888420105
               28 : 0.5970999881625175
               30 : 0.5988999885320664
               31 : 0.6001999858021736
               32 : 0.5996999916434288
               33 : 0.6014999869465828
               34 : 0.6014999872446061
    """
