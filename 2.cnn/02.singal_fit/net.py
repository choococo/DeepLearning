import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, explained_variance_score


# from torch.utils.tensorboard import SummaryWriter

# summaryWriter = SummaryWriter(log_dir="log")


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(1, 256, 3, 1, 0)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(256, 512, 3, 1, 0)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(512, 256, 3, 1, 0)
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
    # net  =Net()
    # x = torch.randn(10, 1, 9)
    # y = net(x)
    # print(y.shape)
    # exit()
    # net = Net(1, 256, 512, 256, 1)
    train_data = np.loadtxt("waveform.data")
    test_data = np.loadtxt("waveform_test.data")
    test_data = torch.FloatTensor(test_data) / 10
    # print(train_data)
    train_data = torch.FloatTensor(train_data) / 10
    # print(train_data)
    net = Net()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    train = True
    index = 160
    if os.path.exists(f"params/{index}.t"):
        net.load_state_dict(torch.load(f"params/{index}.t"))
        print("loaded params success.")

    # if train:
    #     count_list = []
    #     train_loss_list = []
    #     net.train()
    #     for epoch in range(40):
    #         train_loss_list_inter = []
    #         for i in range(len(train_data) - 9):
    #             x = train_data[i:i + 9]
    #             y = train_data[i + 9:i + 10]
    #             x = x.reshape(-1, 1, 9)
    #             y = y.reshape(-1, 1)
    #
    #             out = net(x)
    #
    #             loss = loss_func(out, y)
    #             print(loss.item())
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             train_loss_list_inter.append(loss.item())
    #
    #             # print(loss.item())
    #         train_loss_list.append(np.mean(train_loss_list_inter))
    #         # summaryWriter.add_scalar("Loss", np.mean(train_loss_list_inter), (epoch + 1))
    #         torch.save(net.state_dict(), f"params/{index + epoch + 1}.t")
    # net.load_state_dict(torch.load(f"params/{index}.t"))
    net.eval()
    label = []
    output = []
    count = []
    plt.ion()
    for i in range(len(test_data) - 9):
        x = test_data[i:i + 9]
        y = test_data[i + 9:i + 10]
        x = x.reshape(-1, 1, 9)
        y = y.reshape(-1, 1)
        out = net(x)
        loss = loss_func(out, y)
        # print(loss.item())
        label.append((y.numpy() * 10).reshape(-1))
        output.append((out.detach().numpy() * 10).reshape(-1))
        count.append(i)
        plt.clf()
        plt.plot(count, label, c="blue")
        plt.plot(count, output, c="orange")
        plt.pause(0.01)
    plt.ioff()
    plt.show()

    r2 = r2_score(label, output)
    variance = explained_variance_score(label, output)
    print(r2)
    print(variance)
# summaryWriter.close()
