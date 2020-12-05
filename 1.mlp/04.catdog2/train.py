import torch
import torch.nn as nn
import numpy as np
from data_set import CatDogDataset
from torch.utils.data import DataLoader
from net import CatDogNet
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    BATCH_SIZE = 100
    root = r"F:\2.Dataset\cat_dog"
    train_dataset = CatDogDataset(root, 0)
    val_dataset = CatDogDataset(root, 1)
    test_dataset = CatDogDataset(root, 2)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    net = CatDogNet().to(DEVICE)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    last_acc = 0
    index = 16

    if os.path.exists(f"params/{index}.t"):
        net.load_state_dict(torch.load(f"params/{index}.t"))
        print("loaded params success.")
        net.eval()
        acc_list = []
        for i, (test_x, test_y) in enumerate(test_loader):
            test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
            out = net(test_x)
            test_idx = torch.argmax(out, dim=1)
            accuracy = (torch.sum(torch.eq(test_idx, test_y)) / BATCH_SIZE).item()
            acc_list.append(accuracy)
        last_acc = np.mean(acc_list)
        print("上一次的平均精度：", last_acc)
        # exit(0)

    train_count = 0

    train_count_list = []
    train_loss_list = []
    train_acc_list = []
    val_count = 0
    val_count_list = []
    val_loss_list = []
    val_acc_list = []
    epoch_list = []
    # plt.ion()  # 打开实时画图
    for epoch in range(40):
        train_acc_list_inter = []
        train_loss_list_inter = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = net(x)
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx = torch.argmax(out, dim=1)
            accuracy = (torch.sum(torch.eq(idx, y)) / BATCH_SIZE).item()

            train_count += 1
            train_count_list.append(train_count)
            train_loss_list_inter.append(loss.item())
            train_acc_list_inter.append(accuracy)
        train_acc_list.append(np.mean(train_acc_list_inter))
        train_loss_list.append(np.mean(train_loss_list_inter))
        print(f"{epoch + index + 1} | train_acc：{np.mean(train_acc_list_inter)}")

        val_acc_list_inter = []
        val_loss_list_inter = []
        net.eval()
        for i, (val_x, val_y) in enumerate(val_loader):
            val_x, val_y = val_x.to(DEVICE), val_y.to(DEVICE)
            val_out = net(val_x)
            loss = loss_func(val_out, val_y)
            val_idx = torch.argmax(val_out, dim=1)
            val_accuracy = (torch.sum(torch.eq(val_idx, val_y)) / BATCH_SIZE).item()

            val_count += 1
            val_count_list.append(val_count)
            val_loss_list_inter.append(loss.item())
            # val_acc_list.append(val_accuracy)
            val_acc_list_inter.append(val_accuracy)
        # print("验证集平均精度：", np.mean(acc_list))
        print(f"{epoch + index + 1} | val_acc：{np.mean(val_acc_list_inter)}")
        val_acc_list.append(np.mean(val_acc_list_inter))
        val_loss_list.append(np.mean(val_loss_list_inter))
        last_acc = last_acc
        net.eval()
        acc_list = []
        for i, (test_x, test_y) in enumerate(test_loader):
            test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
            out = net(test_x)
            test_idx = torch.argmax(out, dim=1)
            accuracy = (torch.sum(torch.eq(test_idx, test_y)) / BATCH_SIZE).item()
            acc_list.append(accuracy)

        print(f"{epoch + index + 1} | test_acc：{np.mean(acc_list)}")
        if np.mean(acc_list) > last_acc:
            torch.save(net.state_dict(), f"params/{epoch + index + 1}.t")
            print("params save success!")
            last_acc = np.mean(acc_list)
        epoch_list.append((epoch + index + 1))
        print()
    plt.ion()
    plt.subplot(121)
    plt.title("Accuracy")
    plt.plot(epoch_list, train_acc_list, c="orange", label="train")
    plt.plot(epoch_list, val_acc_list, label="val")
    plt.legend()

    plt.subplot(122)
    plt.title("Loss")
    plt.plot(epoch_list, train_loss_list, marker=">", c="orange", label="train")
    plt.plot(epoch_list, val_loss_list, marker=">", label="val")
    plt.legend()

    # plt.ioff()  # 关闭实时画图
    plt.savefig(f"{index}.jpg")
    plt.pause(0.1)
    plt.ioff()
    plt.close()

"""
        10 : 0.679444432258606
        15 : 0.6822222106986575
"""