import torch
import torch.nn as nn

"""
二维卷积：用于计算机视觉
        二维卷积的形状是[N, C, H, W]
"""
a = torch.randn(100, 1, 20, 20) # [N, C, H, W]
conv_2d = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=2, padding=1)

b = conv_2d(a)
print(b)
print(b.shape)

