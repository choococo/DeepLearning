import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, in_c1, out_c1, out_c2, out_c3, out_c4):
        super().__init__()
        self.conv_1d = nn.Sequential(
            nn.Conv1d(in_c1, out_c1, 3, 1, 0),
            nn.InstanceNorm1d(out_c1),
            nn.ReLU()
        )
        self.conv_2d = nn.Sequential(
            nn.Conv1d(out_c1, out_c2, 3, 1, 0),
            nn.InstanceNorm1d(out_c2),
            nn.ReLU()
        )
        self.conv_3d = nn.Sequential(
            nn.Conv1d(out_c2, out_c3, 3, 1, 0),
            nn.InstanceNorm1d(out_c3),
            nn.ReLU()
        )
        self.fc = nn.Linear(out_c3 * 3, out_c4)

    def forward(self, x):
        y = self.conv_1d(x)
        y = self.conv_2d(y)
        y = self.conv_3d(y)
        y = y.reshape(y.size(0), -1)
        y = self.fc(y)
        return y


if __name__ == '__main__':

    net = Net(1, 256, 512, 256, 1)
    net.load_state_dict(torch.load("./params.pth"))
    train_data = torch.load("./train.data")
    test_data = torch.load("./test.data")
    print(train_data.shape)
    print(test_data.shape)
    loss_func = nn.MSELoss()
    optim = torch.optim.Adam(net.parameters())
    # exit()
    train = 0
    if train:
        net.train()
        for epoch in range(10):
            for i in range(len(train_data) - 9):
                x = train_data[i:i + 9]
                y = train_data[i + 9:i + 10]
                x = x.reshape(-1, 1, 9)
                y = y.reshape(-1, 1)
                out = net(x)
                loss = loss_func(out, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                print("Epoch:{},Loss:{:.3f}".format(epoch, loss.item()))
            torch.save(net.state_dict(), "./params.pth")
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
        print(loss.item())
        label.append(y.numpy().reshape(-1))
        output.append(out.data.numpy().reshape(-1))
        count.append(i)
        plt.clf()
        label_icon, = plt.plot(count, label, linewidth=1, color="blue")
        output_icon, = plt.plot(count, output, linewidth=1, color="red")
        plt.legend([label_icon, output_icon], ["label", "output"], loc="upper right", fontsize=10)

        plt.pause(0.01)
    plt.savefig("./img.pdf")
    plt.ioff()
    plt.show()
    # print(np.shape(label))
    # print(np.shape(output))
    r2 = r2_score(label, output)
    variance = explained_variance_score(label, output)
    print(r2)
    print(variance)
