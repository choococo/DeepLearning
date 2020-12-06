import torch
import torch.nn as nn
from net import CatDogNet
from MyDataset import CatDogDataset
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train:

    def __init__(self, root, batch_size=10, index=0, epoch=10):
        print("Train init...")
        self._summary_writer = r"summaryWriter.txt"
        self._root = root
        self._BATCH_SIZE = batch_size
        self._index = index
        self._EPOCH = epoch

        self._train_dataset = CatDogDataset(self._root, 0)
        self._val_dataset = CatDogDataset(self._root, 1)
        self._test_dataset = CatDogDataset(self._root, 2)

        self._train_loader = DataLoader(self._train_dataset, self._BATCH_SIZE, shuffle=True)
        self._val_loader = DataLoader(self._val_dataset, self._BATCH_SIZE, shuffle=True)
        self._test_loader = DataLoader(self._test_dataset, self._BATCH_SIZE, shuffle=True)

        self._net = CatDogNet().to(DEVICE)
        self._loss_func = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=1e-3)

        self._last_acc = 0
        self._save_params = f"params/{self._index}.t"
        if not os.path.exists("params"):
            os.mkdir("params")

        self._acc_list = []
        if os.path.exists(self._save_params):
            self._net.load_state_dict(torch.load(self._save_params))
            print("loaded params success.")
            self._net.eval()
            for i, (test_x, test_y) in enumerate(self._test_loader):
                test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
                test_out = self._net(test_x)
                test_idx = torch.argmax(test_out, dim=1)
                accuracy = (torch.sum(torch.eq(test_idx, test_y)) / BATCH_SIZE).item()
                self._acc_list.append(accuracy)
            self._last_acc = np.mean(self._acc_list)
            print("last accuracy is : ", self._last_acc)

        self._train_loss_list = []
        self._train_acc_list = []

        self._val_loss_list = []
        self._val_acc_list = []

        self._epoch_list = []
        print("Train has inited.")

    def __call__(self):
        print("Training...")
        for epoch in range(self._EPOCH):
            train_acc_list_inter = []
            train_loss_list_inter = []
            for i, (x, y) in enumerate(self._train_loader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = self._net(x)
                loss = self._loss_func(out, y)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                idx = torch.argmax(out, dim=1)
                accuracy = (torch.sum(torch.eq(idx, y)) / BATCH_SIZE).item()

                train_loss_list_inter.append(loss.item())
                train_acc_list_inter.append(accuracy)
            self._train_acc_list.append(np.mean(train_acc_list_inter))
            self._train_loss_list.append(np.mean(train_loss_list_inter))
            print(f"{epoch + self._index + 1} | train_acc：{np.mean(train_acc_list_inter)}")

            val_acc_list_inter = []
            val_loss_list_inter = []
            self._net.eval()
            for i, (val_x, val_y) in enumerate(self._val_loader):
                val_x, val_y = val_x.to(DEVICE), val_y.to(DEVICE)
                val_out = self._net(val_x)
                loss = self._loss_func(val_out, val_y)
                val_idx = torch.argmax(val_out, dim=1)
                val_accuracy = (torch.sum(torch.eq(val_idx, val_y)) / BATCH_SIZE).item()
                val_loss_list_inter.append(loss.item())
                val_acc_list_inter.append(val_accuracy)
            print(f"{epoch + self._index + 1} | val_acc：{np.mean(val_acc_list_inter)}")
            self._val_acc_list.append(np.mean(val_acc_list_inter))
            self._val_loss_list.append(np.mean(val_loss_list_inter))

            last_acc = self._last_acc
            self._net.eval()
            test_acc_list = []
            for i, (test_x, test_y) in enumerate(self._test_loader):
                test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
                out = self._net(test_x)
                test_idx = torch.argmax(out, dim=1)
                accuracy = (torch.sum(torch.eq(test_idx, test_y)) / BATCH_SIZE).item()
                test_acc_list.append(accuracy)

            print(f"{epoch + self._index + 1} | test_acc：{np.mean(test_acc_list)}")
            self._summary_writer_acc_loss_etc(train_acc_list_inter, val_acc_list_inter, test_acc_list)
            if np.mean(test_acc_list) > last_acc:
                torch.save(self._net.state_dict(), f"params/{epoch + self._index + 1}.t")
                print("params save success!")
                self._last_acc = np.mean(test_acc_list)
            self._epoch_list.append((epoch + self._index + 1))
            print()
        self._draw_acc_loss_figure()
        print("Train has finished")

    def _read_summary_writer_len(self):  # 读取收集精度/损失的数据，用于继续追加
        if isinstance(self._summary_writer, str):
            with open(self._summary_writer, "r") as f:
                length = len(f.readlines())
            return length

    def _draw_acc_loss_figure(self):
        # plt.ion()
        plt.subplot(121)
        plt.title("Accuracy")
        plt.plot(self._epoch_list, self._train_acc_list, c="orange", label="train")
        plt.plot(self._epoch_list, self._val_acc_list, label="val")
        plt.legend()

        plt.subplot(122)
        plt.title("Loss")
        plt.plot(self._epoch_list, self._train_loss_list, marker=">", c="orange", label="train")
        plt.plot(self._epoch_list, self._val_loss_list, marker=">", label="val")
        plt.legend()

        # plt.ioff()  # 关闭实时画图
        plt.savefig(f"{self._index}.jpg")
        plt.show()
        # plt.pause(0.1)
        # plt.ioff()
        # plt.close()

    def _summary_writer_acc_loss_etc(self, train_acc_list_inter, val_acc_list_inter, test_acc_list):
        if isinstance(self._summary_writer, str):
            with open(self._summary_writer, "a+") as f:
                f.write(f"{self._save_params} {np.mean(train_acc_list_inter)} {np.mean(val_acc_list_inter)} {np.mean(test_acc_list)}")
                f.write("\n")
                f.flush()

    def model_params_test(self):
        """
        查看模型的信息
        :return: 输出模型信息
        """
        summary(model=self._net, input_size=(3, 100, 100))


if __name__ == '__main__':
    BATCH_SIZE = 100
    EPOCH = 20
    root = r"F:\2.Dataset\cat_dog"
    index = 1

    train = Train(root=root, batch_size=BATCH_SIZE, index=index, epoch=EPOCH)
    # train.model_params_test()
    train()