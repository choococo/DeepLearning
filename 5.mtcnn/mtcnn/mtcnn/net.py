import torch
import torch.nn as nn

"""
网络的视频
"""


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1, 0),
            nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # 这里使用的重叠池化，比一般的池化更加严重
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 16, 3, 1, 0),
            nn.BatchNorm2d(16),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.conv4 = nn.Conv2d(32, 1, 1, 1)
        self.conv5 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        confidence = torch.sigmoid(self.conv4(out))  # 后面要使用BCELoss需要对数据做归一化
        offset = self.conv5(out)
        return confidence, offset


class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1, 1),
            nn.BatchNorm2d(28),
            nn.PReLU(),
            # nn.MaxPool2d(kernel_size=(3, 3), stride=2)
            # 卷积代替池化
            nn.Conv2d(28, 28, 3, 2, 0),
            nn.BatchNorm2d(28),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(28, 48, 3, 1, 0),  # 这里需要手动调整一下
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)  # 这里使用的重叠池化，比一般的池化更加严重
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, 2, 1, 0),  # 这里是2x2的卷积，需要注意
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU()
        )
        self.out1 = nn.Linear(128, 1)
        self.out2 = nn.Linear(128, 4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.reshape(x.size(0), -1)
        out = self.fc1(out)
        confidence = torch.sigmoid(self.out1(out))  # 后面要使用BCELoss需要对数据做归一化
        offset = self.out2(out)
        return confidence, offset


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            # nn.MaxPool2d(kernel_size=(3, 3), stride=2)
            nn.Conv2d(32, 32, 3, 2, 0),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 0),  # 这里需要手动调一下
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # nn.MaxPool2d(kernel_size=(3, 3), stride=2)
            nn.Conv2d(64, 64, 3, 2, 0),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            # 这里使用的是2x2的卷积核，看情况是否进行修改为3x3的卷积核
            nn.Conv2d(64, 64, 2, 2, 0),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 2, 1, 0),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.PReLU()
        )
        self.out1 = nn.Linear(256, 1)
        self.out2 = nn.Linear(256, 14)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.reshape(x.size(0), -1)
        out = self.fc(out)
        confidence = torch.sigmoid(self.out1(out))  # 后面要使用BCELoss需要对数据做归一化
        offset = self.out2(out)
        return confidence, offset


if __name__ == '__main__':
    x = torch.randn(2, 3, 12, 12)

    p_net = PNet()
    conf, off = p_net(x)
    print(conf.shape)
    print(off.shape)

    x = torch.randn(2, 3, 24, 24)
    r_net = RNet()
    conf, off = r_net(x)
    print(conf.shape)
    print(off.shape)

    x = torch.randn(2, 3, 48, 48)
    o_net = ONet()
    conf, off = o_net(x)
    print(conf.shape)
    print(off.shape)
