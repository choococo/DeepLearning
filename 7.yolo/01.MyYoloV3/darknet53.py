import torch
import torch.nn as nn
import torch.nn.functional as F
from FRN import FilterResponseNorma2d


# 进行上采样：采用邻近上采样，速度快
class UpSamplingLayer(nn.Module):

    def __init__(self):
        super(UpSamplingLayer, self).__init__()

    @staticmethod
    def forward(x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


# CBL卷积层，对CNN+BN+LeakyRelu进行封装
class ConvolutionalLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            # nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(0.1)
            FilterResponseNorma2d(out_channels)
        )

    def forward(self, x):
        return self.sub_module(x)


# 残差快：可以让网络变得比较深，增加网络的抽象能力
class ResidualLayer(nn.Module):

    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, 1, 1, 0),
            ConvolutionalLayer(in_channels // 2, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.sub_module(x) + x


# 下采样层：利用卷积+stride=2进行下采样
class DownSamplingLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownSamplingLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)


# 卷积集模块：是一种自监督的方式，让13x13、26x26、52x52变得不同，相当于是从上一层得到的特征进行再一次的特征提取
# 采用增加信息融合的方式进行监督
class ConvolutionalSet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.sub_module(x)


class MainNet(nn.Module):

    def __init__(self, cls_num):
        super(MainNet, self).__init__()
        self.trunk_52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownSamplingLayer(32, 64),
            ResidualLayer(64),  # 1
            DownSamplingLayer(64, 128),
            ResidualLayer(128),  # 2
            ResidualLayer(128),
            DownSamplingLayer(128, 256),
            ResidualLayer(256),  # 8
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256)
        )

        self.trunk_26 = nn.Sequential(
            DownSamplingLayer(256, 512),
            ResidualLayer(512),  # 8
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512)
        )

        self.trunk_13 = nn.Sequential(
            DownSamplingLayer(512, 1024),
            ResidualLayer(1024),  # 4
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024)
        )
        self.conv_set_13 = nn.Sequential(
            ConvolutionalSet(1024, 512)
        )
        self.detection_13 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 3 * (5 + cls_num), 1, 1, 0)
        )

        self.up_26 = nn.Sequential(
            ConvolutionalLayer(512, 256, 1, 1, 0),
            UpSamplingLayer()
        )
        self.conv_set_26 = nn.Sequential(
            ConvolutionalSet(768, 256)
        )
        self.detection_26 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, 3 * (5 + cls_num), 1, 1, 0)
        )

        self.up_52 = nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 0),
            UpSamplingLayer()
        )
        self.conv_set_52 = nn.Sequential(
            ConvolutionalSet(384, 128)
        )
        self.detection_52 = nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256, 3 * (5 + cls_num), 1, 1, 0)
        )

    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)
        con_set_out_13 = self.conv_set_13(h_13)
        detection_13 = self.detection_13(con_set_out_13)

        up_out_26 = self.up_26(con_set_out_13)
        route_out_26 = torch.cat([up_out_26, h_26], dim=1)  # [2, 768, 26, 26]
        con_set_out_26 = self.conv_set_26(route_out_26)  # [2, 256, 26, 26]
        detection_26 = self.detection_26(con_set_out_26)

        up_out_52 = self.up_52(con_set_out_26)  # [2, 128, 52, 52]
        route_out_52 = torch.cat([up_out_52, h_52], dim=1)  # [2, 384, 52, 52]
        con_set_out_52 = self.conv_set_52(route_out_52)  # [2, 128, 52, 52]
        detection_52 = self.detection_52(con_set_out_52)
        return detection_13, detection_26, detection_52


if __name__ == '__main__':
    a = torch.randn(2, 3, 416, 416)
    net = MainNet(3)
    # net.eval()
    # y1, y2, y3 = net(x)
    # print(y1.shape)
    # print(y2.shape)
    # print(y3.shape)
    from torchsummary import summary
    summary(net.cuda(), input_size=(3, 416, 416))
