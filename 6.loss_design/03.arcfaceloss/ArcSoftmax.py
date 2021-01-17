import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class ArcNet(nn.Module):
    def __init__(self, feature_dim=2, cls_dim=10):
        super().__init__()
        # 生成一个隔离带向量，训练这个向量和原来的特征向量尽量分开，达到增加角度的目的
        self.W = nn.Parameter(torch.randn(feature_dim, cls_dim).cuda(), requires_grad=True)

    def forward(self, feature, m=0.5, s=10):
        # 对特征维度进行标准化
        x = F.normalize(feature, dim=1)  # shape=【100，2】
        w = F.normalize(self.W, dim=0)  # shape=【2，10】
        # s=64
        s = torch.sqrt(torch.sum(torch.pow(x, 2))) * torch.sqrt(torch.sum(torch.pow(w, 2)))
        # 做L2范数化，将cosa变小，防止acosa梯度爆炸
        cosa = torch.matmul(x, w) / s
        # print(cosa)#[-1,1]

        "标准化后的x(-1,1)再求平方(1,1)，相当于求它的单位向量(1)，所以求x的平方和就是批次100*1=100"
        "同理标准化后的w有10个维度，就等于10*1=10"
        "所以s就等于sqrt(100)*sqrt(10)≈31.6"
        a = torch.acos(cosa)  # 反三角函数得出的是弧度，而非角度，1弧度=1*180/3.14=57角度
        # 这里对e的指数cos(a+m)再乘回来，让指数函数的输出更大，
        # 从而使得arcsoftmax输出更小，即log_arcsoftmax输出更小，则-log_arcsoftmax更大。
        # m=0.5
        arcsoftmax = torch.exp(
            s * torch.cos(a + m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
            s * cosa) + torch.exp(s * torch.cos(a + m)))

        return arcsoftmax


"""
    s=64
    s = torch.sqrt(torch.sum(torch.pow(x, 2))) * torch.sqrt(torch.sum(torch.pow(w, 2)))
    做L2范数化，将cosa变小，防止acosa梯度爆炸
    print(arcsoftmax)
    这里arcsomax的概率和不为1，小于1。这会导致交叉熵损失看起来很大，且最优点损失也很大
    print(torch.sum(arcsoftmax, dim=1))
    exit()
    
    lmsoftmax = (torch.exp(cosa) - m) / (
             torch.sum(torch.exp(cosa) - m, dim=1, keepdim=True) - (torch.exp(cosa) - m) + (torch.exp(cosa) - m))
     return lmsoftmax
"""

if __name__ == '__main__':
    arc = ArcNet(feature_dim=2, cls_dim=10)
    feature = torch.randn(100, 2).cuda()
    out = arc(feature)
    print(feature)
    print('out', out)
