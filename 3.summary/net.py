import torch
from torch import nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(3072,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        out = self.layer(x)
        return out
class NetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        out = self.layer(x)
        return out

class NetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1920),
            nn.ReLU(),
            nn.Linear(1920, 1080),
            nn.ReLU(),
            nn.Linear(1080, 1024),
            nn.ReLU(),
            nn.Linear(1024, 720),
            nn.ReLU(),
            nn.Linear(720, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        out = self.layer(x)
        return out
if __name__ == '__main__':

    net = NetV2()
    x = torch.randn(4,3072)
    y = net(x)
    print(y.shape)