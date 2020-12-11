import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, explained_variance_score
from MyDataset import WaveformDataset
from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter

# summaryWriter = SummaryWriter(log_dir="log")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(1, 256, 3, 1, 0),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(256, 512, 3, 1, 0),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(512, 256, 3, 1, 0),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(3 * 256, 1)
        )

    def forward(self, x):
        y = self.conv_layer1(x)
        y = self.conv_layer2(y)
        y = self.conv_layer3(y)
        # print(y.shape)
        y = y.reshape(x.size(0), -1)
        y = self.fc_layer(y)
        return y


if __name__ == '__main__':
    EPOCH = 20
    train_dataset = WaveformDataset()
    test_dataset = WaveformDataset()

    based_data = train_dataset.based_data
    based_data_list = [] # 基准数据
    for i in range(30):
        based_data_list.extend(based_data)

    train_data = train_dataset.train_data

    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)
    index = 20
    save_params = f"params/{index}.t"
    net = Net().to(DEVICE)
    if os.path.exists(save_params):
        net.load_state_dict(torch.load(save_params))
        print("loaded params success.")

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    Train = False
    if Train:
        for epoch in range(EPOCH):
            for iter in range(30):
                for i, (x, y) in enumerate(train_loader):
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    x = x[:, None, :]
                    out = net(x) # [N, 1, 9] -> [N, 1]
                    loss = loss_func(out, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            if epoch > 10:
                torch.save(net.state_dict(), f"params/{epoch + index + 1}.t")
    else:
        y_list = []
        out_list = []
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x[:, None, :]
            out = net(x)  # [N, 1, 9] -> [N, 1]
            out = out.reshape(-1).cpu().detach().numpy()
            out_list.extend(out)
            y_list.extend(y.cpu().detach().numpy())

        plt.plot([i for i in range(291)], based_data_list[9:])
        # plt.plot([i for i in range(300)], train_data)
        # plt.plot([i for i in range(291)], y_list)
        plt.plot([i for i in range(291)], out_list)
        plt.show()
        print(r2_score(y_list, out_list))
