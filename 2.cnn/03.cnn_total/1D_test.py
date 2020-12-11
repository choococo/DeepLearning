import torch
import torch.nn as nn

"""
一维卷积：用于序列模型、信号处理、自然语言处理
        一维卷积的形状是:[N, C, L]
"""
a = torch.randn(100, 1, 20)  # [N,C,L]\
conv_1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)

b = conv_1d(a)
print(b)

print(b.shape)
