import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)


if __name__ == '__main__':
    net = Net()
    x = torch.tensor([[0.4989, -1.8265, 0.7443],
                      [-0.3545, -0.5649, -0.6938],
                      [0.4845, -1.4270, -1.7665]])
    target = torch.tensor([0, 1, 2])

    loss_func2 = torch.nn.CrossEntropyLoss()
    loss2 = loss_func2(x, target)
    print(loss2)
    print("=========================")
    x = torch.tensor([[0.4989, -1.8265, 0.7443],
                      [-0.3545, -0.5649, -0.6938],
                      [0.4845, -1.4270, -1.7665]])
    # out = torch.softmax(x, dim=1)
    # out = torch.log(out)

    out = torch.log_softmax(x, dim=1)
    print(out)
    # loss_func = torch.nn.NLLLoss()
    # loss = loss_func(out, target)
    loss = torch.tensor((0.8654+1.1357+2.4767) / 3)
    print(loss)
