# import torch
# import torch.nn as nn
# from torchsummary import summary
#
#
# class Convolutional(nn.Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
#         super(Convolutional, self).__init__()
#         self.sub_module = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.sub_module(x)
#
#
# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1_layer = nn.Sequential(
#             # 因为有的小黄人会出现在边缘，因此对加入padding，增加对边缘的提取能力
#             Convolutional(3, 16, 3, 1, 2),  # [2, 16, 226, 226]
#             nn.MaxPool2d(2)  # [2, 16, 113, 113]
#         )
#         self.conv2_layer = nn.Sequential(
#             Convolutional(16, 32, 3, 1, 2),  # [2, 32, 115, 115]
#             nn.MaxPool2d(2)  # [2, 32, 57, 57]
#         )
#         self.conv3_layer = nn.Sequential(
#             Convolutional(32, 64, 3, 1, 2),  # [2, 64, 59, 59]
#             nn.MaxPool2d(2)  # [2, 64, 29, 29]
#         )
#         self.conv4_layer = nn.Sequential(
#             Convolutional(64, 128, 3, 1, 2),  # [2, 128, 31, 31]
#             nn.MaxPool2d(2)  # [2, 128, 15, 15]
#         )
#         self.fc_layer = nn.Linear(128 * 15 * 15, 5)
#
#     def forward(self, x):
#         out = self.conv1_layer(x)
#         out = self.conv2_layer(out)
#         out = self.conv3_layer(out)
#         out = self.conv4_layer(out)
#         out = out.reshape(x.size(0), -1)
#         out = self.fc_layer(out)
#         return torch.relu(out[:, 0]), out[:, 1:]
#
#
# if __name__ == '__main__':
#     net = Net()
#     x = torch.randn(2, 3, 224, 224)
#     y = net(x)
#     print(y.shape)
#     summary(net.cuda(), (3, 224, 224))

# """
# ================================================================
# Total params: 241,925
# Trainable params: 241,925
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 51.39
# Params size (MB): 0.92
# Estimated Total Size (MB): 52.89
# """
from torch import nn, optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # batch*64*112*112

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # batch*128*56*56

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # batch*256*28*28

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # batch*512*14*14

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # batch*256*7*7

        self.conv56 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # batch*128*3*3

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )  # batch*64*1*1

        self.fcn = nn.Sequential(
            nn.Linear(in_features=64 * 1 * 1, out_features=5),

        )

    def forward(self, x):
        # print(x.type())
        # x = torch.FloatTensor(x)
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)
        y5 = self.conv56(y5)
        y6 = self.conv6(y5)

        y6 = y6.reshape(y6.size(0), -1)
        output = self.fcn(y6)  # [N, 5]

        output1 = output[:, :4] # [N, 4]

        output2 = torch.sigmoid(output[:, 4])  # [N, 1]
        # output2 = output[:, 4:]  # [N, 1]

        return output2, output1


if __name__ == '__main__':
    from torchsummary import summary

    net = Net().cuda()
    summary(net, (3, 224, 224))
