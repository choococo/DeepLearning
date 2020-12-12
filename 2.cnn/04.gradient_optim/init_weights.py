import torch
import torch.nn as nn

"""
对应xmind中的权重初始化：
    一个是Yan Lecun的
    一个是Xavier的
    一个是He KaiMing的
"""
# 加入随机种子，保持当前的随机数不变
seed = torch.manual_seed(0)

x = torch.randn(100, 400)
print(x.mean(), x.var())  # 计算当前的均值和方差

w1 = nn.init.normal_(x, 0.0, 0.0002)
print(w1.mean(), w1.var())

w2 = nn.init.uniform_(x, -0.2, 0.2)
print(torch.min(w2), torch.max(w2))

w3 = nn.init.xavier_normal_(x)
print(w3.mean(), w3.var())

w4 = nn.init.xavier_uniform_(x)
print(torch.min(w4), torch.max(w4))

w5 = nn.init.kaiming_normal_(x)
print(w5.mean(), w5.var())

w6 = nn.init.uniform_(x)
print(torch.min(w6), torch.max(w6))

