import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyDataset
from torch.utils.tensorboard import SummaryWriter
from net import Net
import os
import numpy as np
from torchsummary import summary
import torchvision
from sklearn.metrics import r2_score
from torchvision import models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MobileNet(nn.Module): # 使用mobilenet网络进行学习
    def __init__(self):
        super(MobileNet, self).__init__()
        self.mobile_net = models.mobilenet_v2()
        self.mobile_net.classifier[1] = nn.Linear(1280, 5)

    def forward(self, x):
        out = self.mobile_net(x)
        conf = torch.sigmoid(out[:, 4])
        bbox = out[:, :4]
        return conf, bbox


def _loss_func_total(self, output, target, alpha=1):
    # output [conf, x1, y1, x2, y2] target [conf, x1, y1, x2, y2]
    # loss_conf =  loss_func(output[:, 0:1], target[:, 0:1])
    loss_box = nn.MSELoss()(output[:, 1:], target[:, 1:])
    # loss_total = alpha * loss_conf + (1 - alpha) * loss_box
    loss_total = loss_box
    return loss_total


if __name__ == '__main__':
    BATCH_SIZE = 30
    EPOCH = 50
    # 需要训练的轮次
    root = r"F:\2.Dataset\Yellow\Minions2"
    index = 2  # 权重索引，第一次训练为零

    # summaryWriter = SummaryWriter("./logs")

    train_dataset = MyDataset(root, flag=0)
    val_dataset = MyDataset(root, flag=1)
    test_dataset = MyDataset(root, flag=2)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

    net = Net().to(DEVICE)
    loss_func1 = nn.BCELoss()
    loss_func2 = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    last_acc = 0
    save_params = f"params2/{index}.t"
    if not os.path.exists("params2"):
        os.mkdir("params2")

    train_loss_list = []  # loss
    train_acc_list = []

    val_loss_list = []  # acc
    val_acc_list = []

    train_r2_list = []  # r2
    val_r2_list = []

    r2_max = 0  # 过拟合自动停止参数 r2最大值
    average_num = 5  # 多少轮算一次平均
    count = 0  # 初始化值

    for epoch in range(EPOCH):
        train_acc_list_inter = []
        train_loss_list_inter = []
        train_r2_list_inter = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            conf_out, box_out = net(x)  # [N, 1] [N, 4]
            conf_label = y[:, 0]
            box_label = y[:, 1:]

            loss1 = loss_func1(conf_out, conf_label)
            loss2 = loss_func2(box_out, box_label)
            loss = loss1 + loss2
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            box_out = box_out.cpu().detach().numpy()
            box_label = box_label.cpu().detach().numpy()

            r2 = r2_score(box_label, box_out)

            conf_out[conf_out >= 0.5] = 1.0
            conf_out[conf_out < 0.5] = 0.0
            accuracy = (torch.sum(torch.eq(conf_out, conf_label)) / BATCH_SIZE).item()

            train_r2_list_inter.append(r2)
            train_acc_list_inter.append(accuracy)
            train_loss_list_inter.append(loss.item())
        # train_r2_avg = np.mean(train_r2_list_inter)
        # print("====>", train_r2_avg)

        # train_acc_list.append(np.mean(train_acc_list_inter))
        # train_loss_list.append(np.mean(train_loss_list_inter))
        train_r2_list.append(np.mean(train_r2_list_inter))
        print(
            f"{epoch + index + 1} | train_acc：{np.mean(train_acc_list_inter)} | train_loss: {np.mean(train_loss_list_inter)} | r2_score: {np.mean(train_r2_list_inter)}")

        val_acc_list_inter = []
        val_loss_list_inter = []
        val_r2_list_inter = []
        net.eval()

        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            conf_out, box_out = net(x)  # [N, 1] [N, 4]

            conf_label = y[:, 0]
            box_label = y[:, 1:]

            loss1 = loss_func1(conf_out, conf_label)
            loss2 = loss_func2(box_out, box_label)
            loss = loss1 + loss2
            # print(loss1.item(),loss2.item())
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            box_out = box_out.cpu().detach().numpy()
            box_label = box_label.cpu().detach().numpy()
            conf_out[conf_out >= 0.5] = 1.0
            conf_out[conf_out < 0.5] = 0.0
            val_accuracy = (torch.sum(torch.eq(conf_out, conf_label)) / BATCH_SIZE).item()

            r2 = r2_score(box_label, box_out)
            val_r2_list_inter.append(r2)
            val_loss_list_inter.append(loss.item())
            val_acc_list_inter.append(val_accuracy)

        val_r2_s = np.mean(val_r2_list_inter)
        print("====>", val_r2_s)

        # val_acc_list.append(np.mean(val_acc_list_inter))
        # val_loss_list.append(np.mean(val_loss_list_inter))
        val_r2_list.append(np.mean(val_r2_list_inter))
        print(
            f"{epoch + index + 1} | val_acc  ：{np.mean(val_acc_list_inter)} | val_loss  : {np.mean(val_loss_list_inter)} | r2_score : {np.mean(val_r2_list_inter)}")

        if val_r2_s > r2_max:
            r2_max = val_r2_s
            torch.save(net.state_dict(), f"params2/{epoch + index + 1}.t")

        # 加入过拟合自动判断，让程序过拟合自动停止
        r2_length = len(val_r2_list)
        if (len(val_r2_list) - count) % 10 == 0:
            last_avg_acc = np.mean(val_r2_list[count:r2_length - average_num])
            moment_avg_acc = np.mean(val_r2_list[r2_length - average_num:r2_length])
            if last_avg_acc > moment_avg_acc:
                print(max(val_r2_list))
                exit()
            count += average_num
