from torch import nn
import torch

class Conv_Net_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=28,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=28,out_channels=56,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(56,68,3,2),
            nn.ReLU(),
            nn.Conv2d(68,93,3),
            nn.ReLU(),
            nn.Conv2d(93, 128, 3),
            nn.ReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128*2*2,10),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        conv_out = self.conv_layer(x)
        # NCHW-->NV
        conv_out = conv_out.reshape(-1,128*2*2)
        out = self.fc_layer(conv_out)
        return out

if __name__ == '__main__':
    net = Conv_Net_V1()
    x = torch.randn(4,3,32,32)
    y = net(x)
    print(y.shape)