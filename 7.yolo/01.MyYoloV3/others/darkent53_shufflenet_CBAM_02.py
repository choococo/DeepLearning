import torch
import torch.nn as nn
import torch.nn.functional as F
from FRN import FilterResponseNorma2d


class ChannelAttention(nn.Module):  # 通道数量不变，注意力在通道层面
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        # 自适应的大小，根据需要定，（A,B） -->A*B
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 二元自适应均值汇聚层（自适应池化）1->1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 二元自适应最大值汇聚层

        self.sub_module = nn.Sequential(  # 保持通道不变，增加网络深度，用1*1的卷积核进行卷
            nn.Conv2d(in_channels, in_channels // ratio, 1, 1, 0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, 1, 0, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sub_module(self.avg_pool(x))
        max_out = self.sub_module(self.max_pool(x))

        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # keepdim=True保持维度和原始数据维度相同
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 拿到最大值
        x = torch.cat([avg_out, max_out], dim=1)  # 在通道层面做路由
        x = self.conv1(x)  # 把两个通道变成1个通道
        return self.sigmoid(x)


class ShuffleBlock(torch.nn.Module):

    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        g = self.groups
        return x.reshape(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class UpSampleLayer(torch.nn.Module):

    def __init__(self):
        super(UpSampleLayer, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)


class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels * 2, 1, 1, 0, groups=in_channels),  # 110，和311都不会改变特征图的大小
            ConvolutionalLayer(in_channels * 2, in_channels * 2, 3, 1, 1, groups=in_channels * 2),
            ShuffleBlock(groups=in_channels * 2),
            ConvolutionalLayer(in_channels * 2, in_channels, 1, 1, 0, groups=in_channels)
        )

        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.sub_module(x)
        out = self.ca(out) * out  # 进行广播
        out = self.sa(out) * out  # 进行广播
        return x + out


class DownSamplingLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)


class ConvolutionalSet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0, groups=out_channels),
            ShuffleBlock(groups=out_channels),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0, groups=out_channels),
            ShuffleBlock(groups=out_channels),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0, groups=out_channels),
            ShuffleBlock(groups=out_channels),
        )
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.sub_module(x)
        out = self.ca(out) * out  # 进行广播
        out = self.sa(out) * out  # 进行广播
        # return self.sub_module(x)
        return out  # 这里不使用残差了


class MainNet(torch.nn.Module):

    def __init__(self, cls_num):
        super(MainNet, self).__init__()

        self.trunk_52 = torch.nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            ConvolutionalLayer(32, 64, 3, 2, 1),

            ResidualLayer(64),
            DownSamplingLayer(64, 128),

            ResidualLayer(128),
            ResidualLayer(128),
            DownSamplingLayer(128, 256),

            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
        )

        self.trunk_26 = torch.nn.Sequential(
            DownSamplingLayer(256, 512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
        )

        self.trunk_13 = torch.nn.Sequential(
            DownSamplingLayer(512, 1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024)
        )

        self.conv_set_13 = torch.nn.Sequential(
            ConvolutionalSet(1024, 512)
        )

        self.detection_13 = torch.nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            torch.nn.Conv2d(1024, 3 * (5 + cls_num), 1, 1, 0)
        )

        self.up_26 = torch.nn.Sequential(
            ConvolutionalLayer(512, 256, 1, 1, 0),  #
            UpSampleLayer()  # 13->26
        )

        self.conv_set_26 = torch.nn.Sequential(
            ConvolutionalLayer(768, 256, 1, 1, 0),
        )

        self.detection_26 = torch.nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            torch.nn.Conv2d(512, 3 * (5 + cls_num), 1, 1, 0)
        )

        self.up_52 = torch.nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 0),  #
            UpSampleLayer()
        )

        self.conv_set_52 = torch.nn.Sequential(
            ConvolutionalLayer(384, 128, 1, 1, 0),
        )

        self.detection_52 = torch.nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            torch.nn.Conv2d(256, 3 * (5 + cls_num), 1, 1, 0)
        )

    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        conv_set_out_13 = self.conv_set_13(h_13)
        detection_out_13 = self.detection_13(conv_set_out_13)

        up_out_26 = self.up_26(conv_set_out_13)
        route_out_26 = torch.cat((up_out_26, h_26), dim=1)
        conv_set_out_26 = self.conv_set_26(route_out_26)
        detection_out_26 = self.detection_26(conv_set_out_26)

        up_out_52 = self.up_52(conv_set_out_26)
        route_out_52 = torch.cat((up_out_52, h_52), dim=1)
        conv_set_out_52 = self.conv_set_52(route_out_52)
        detection_out_52 = self.detection_52(conv_set_out_52)

        return detection_out_13, detection_out_26, detection_out_52


if __name__ == '__main__':
    net = MainNet(5)
    # net.cuda().half()
    # a = torch.cuda.HalfTensor(2, 3, 416, 416)
    # y_13, y_26, y_52 = net(a)
    # print(y_13.shape)
    # print(y_26.shape)
    # print(y_52.shape)
    from torchsummary import summary

    summary(net.cuda(), input_size=(3, 416, 416))

    """
    Total params: 24,563,566
    Trainable params: 24,563,566
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 1.98
    Forward/backward pass size (MB): 3167.03
    Params size (MB): 93.70
    Estimated Total Size (MB): 3262.71
    """
