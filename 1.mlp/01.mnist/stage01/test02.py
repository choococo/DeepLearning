import torch

x = torch.randn([2, 8]) # 生成正态分布的数据
print(x.shape)
# w = torch.randn([8, 4])
# print(torch.mm(x, w))
# exit(0)

w = torch.nn.Parameter(torch.randn(1)) # 生成随机权重
print(w)


w = torch.nn.Linear(8, 4) # 生成线性值
# print(list(w.parameters())[0], list(w.parameters())[1]) # 打印权重参数
print(list(w.parameters())[0].shape, list(w.parameters())[1].shape)
print(w.weight.shape, w.bias.shape) # 查看权重参数
print(torch.mm(x, w.weight.T) + w.bias) # 这个是记性计算
