import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from TrainTemp.dataset import MyDataset
from torch.utils.tensorboard import SummaryWriter
from logger.loggers import classLogger, funcLogger
from TrainTemp.net import NetTempV1
import os
import numpy as np
from TrainTemp.utils import draw_acc_loss_figure
from torchsummary import summary

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# tensorboard --logdir=logs
@classLogger
class Train(object):

    def __init__(self, root=None, batch_size=100, index=0, epoch=10, is_train=True):

        self._summaryWriter = SummaryWriter("./logs")
        self._root = root
        self._BATCH_SIZE = batch_size
        self._index = index
        self._EPOCH = epoch
        print(self._EPOCH)

        self._train_dataset = MyDataset(self._root, flag=0)
        self._val_dataset = MyDataset(self._root, flag=1)
        self._test_dataset = MyDataset(self._root, flag=2)

        self._train_loader = DataLoader(self._train_dataset, self._BATCH_SIZE, shuffle=True)
        self._val_loader = DataLoader(self._val_dataset, self._BATCH_SIZE, shuffle=True)
        self._test_loader = DataLoader(self._test_dataset, self._BATCH_SIZE, shuffle=True)

        self._net = NetTempV1().to(DEVICE)
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
                accuracy = (torch.sum(torch.eq(test_idx, test_y)) / self._BATCH_SIZE).item()
                self._acc_list.append(accuracy)
            self._last_acc = np.mean(self._acc_list)

            print("last accuracy is : ", self._last_acc)
            if is_train:
                print(" ")
            else:
                exit()

        self._train_loss_list = []
        self._train_acc_list = []

        self._val_loss_list = []
        self._val_acc_list = []

        self._epoch_list = []

    @funcLogger
    def __call__(self, is_matplotlib_draw=False):
        for epoch in range(self._EPOCH):
            self._train(epoch)
            self._val(epoch)
            self._test(epoch)
            train_avg_loss = self._train_loss_list[epoch]
            val_avg_loss = self._val_loss_list[epoch]
            train_avg_score = self._train_acc_list[epoch]
            val_avg_score = self._val_acc_list[epoch]
            # 收集多个标量
            self._summaryWriter.add_scalars("loss", {"train_loss": train_avg_loss, "val_loss": val_avg_loss}, epoch)
            self._summaryWriter.add_scalars("acc", {"train_score": train_avg_score, "val_score": val_avg_score}, epoch)
            # 收集单个标量
            # self._summaryWriter.add_scalar("score", val_avg_score, epoch)

            if is_matplotlib_draw:
                # index, epoch_list, train_acc_list, val_acc_list, train_loss_list, val_loss_list
                draw_acc_loss_figure(self._index, self._epoch_list,
                                     self._train_acc_list, self._val_acc_list,
                                     self._train_loss_list, self._val_loss_list)

        self._summaryWriter.close()

    def _train(self, epoch):
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
            accuracy = (torch.sum(torch.eq(idx, y)) / self._BATCH_SIZE).item()

            train_loss_list_inter.append(loss.item())
            train_acc_list_inter.append(accuracy)
        self._train_acc_list.append(np.mean(train_acc_list_inter))
        self._train_loss_list.append(np.mean(train_loss_list_inter))
        print(
            f"{epoch + self._index + 1} | train_acc：{np.mean(train_acc_list_inter)} | train_loss: {np.mean(train_loss_list_inter)}")

    def _val(self, epoch):
        val_acc_list_inter = []
        val_loss_list_inter = []
        self._net.eval()
        for i, (val_x, val_y) in enumerate(self._val_loader):
            val_x, val_y = val_x.to(DEVICE), val_y.to(DEVICE)
            val_out = self._net(val_x)
            loss = self._loss_func(val_out, val_y)
            val_idx = torch.argmax(val_out, dim=1)
            val_accuracy = (torch.sum(torch.eq(val_idx, val_y)) / self._BATCH_SIZE).item()
            val_loss_list_inter.append(loss.item())
            val_acc_list_inter.append(val_accuracy)
        self._val_acc_list.append(np.mean(val_acc_list_inter))
        self._val_loss_list.append(np.mean(val_loss_list_inter))
        print(
            f"{epoch + self._index + 1} | val_acc  ：{np.mean(val_acc_list_inter)} | val_loss  : {np.mean(val_loss_list_inter)}")

    def _test(self, epoch):
        last_acc = self._last_acc
        self._net.eval()
        test_acc_list = []
        for i, (test_x, test_y) in enumerate(self._test_loader):
            test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
            out = self._net(test_x)
            test_idx = torch.argmax(out, dim=1)
            accuracy = (torch.sum(torch.eq(test_idx, test_y)) / self._BATCH_SIZE).item()
            test_acc_list.append(accuracy)

        print(f"{epoch + self._index + 1} | test_acc：{np.mean(test_acc_list)}")
        if np.mean(test_acc_list) > last_acc:
            torch.save(self._net.state_dict(), f"params/{epoch + self._index + 1}.t")
            print("params save success!")
            self._last_acc = np.mean(test_acc_list)
        self._epoch_list.append((epoch + self._index + 1))
        print()

    def model_params_test(self, input_size):
        """
        查看模型的信息
        :param input_size: 输入形状
        :return: 输出模型信息
        """
        summary(model=self._net, input_size=input_size)

# if __name__ == '__main__':
#     BATCH_SIZE = 100
#     EPOCH = 35
#     root = r"F:\2.Dataset\cat_dog"
#     index = 0
#     train = Train(root=root, batch_size=BATCH_SIZE, index=index, epoch=EPOCH, is_train=False)
#     train()
#     # train.model_params_test(input_size=(3, 100, 100))
