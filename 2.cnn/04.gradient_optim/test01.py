import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.W = nn.Parameter(torch.randn(784, 10))

    def forward(self, x):
        h = x @ self.W  # @ 表示矩阵乘法
        # Softmax
        h = torch.exp(h)
        z = torch.sum(h, dim=1, keepdim=True) # 保持原来的维度
        return h / z


if __name__ == '__main__':
    net = Net()
    x = torch.randn(6, 784)
    y = net(x)
    print(y.shape)

