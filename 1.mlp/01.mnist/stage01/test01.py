import torch
import torchvision

x = torch.randn([3, 4], requires_grad=True) # 定义一个张量，有梯度的张量
print(x) # 输出x
print(x.detach().numpy()) # detach() 取多个张量值，并转成numpy数组

# 定义一个标量的张量，类型为float32, 有梯度
x = torch.tensor(1, dtype=torch.float32, requires_grad=True)
print(x.detach().numpy()) # 取出值
print(x.item()) # 取出标量值，当只有一个值的时候，可以取
