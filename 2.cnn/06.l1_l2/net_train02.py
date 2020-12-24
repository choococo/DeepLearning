import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils import data
import matplotlib.pyplot as plt
from test04.dataset_sampling import MyDataset
# from tensorboardX import SummaryWriter
from torchvision import datasets, transforms


# writer = SummaryWriter("./logs")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.Dropout(0.5),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(14 * 14 * 64, 128),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, epoch):
        y = torch.dropout(self.conv(x), 0.3, True)
        y = y.reshape(y.size(0), -1)
        y = torch.dropout(self.fc1(y), 0.3, True)
        y = self.fc2(y)
        return y


if __name__ == '__main__':
    batch_size = 100
    # data_path = r"E:\datasets\MNIST_IMG"
    save_params = "./net_params.pth"
    save_net = "./net.pth"
    # train_data = MyDataset(data_path,True)
    # test_data = MyDataset(data_path,False)
    train_data = datasets.MNIST("../data", True, transforms.ToTensor(), download=True)
    test_data = datasets.MNIST("../data", False, transforms.ToTensor(), download=False)
    print(train_data.data.shape)
    train_loader = data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Net().to(device)
    # net.load_state_dict(torch.load(save_params))
    # net = torch.load(save_net).to(device)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                             weight_decay=0)

    net.train()

    for epoch in range(1):
        train_loss = 0
        train_acc = 0
        alpha = 0.01
        gamma = 0.2
        for i, (x, y) in enumerate(train_loader):
            # y=torch.zeros(y.size(0),10).scatter_(1,y.reshape(-1,1),1)
            x = x.to(device)
            y = y.to(device)
            out = net(x, epoch)
            L1 = 0
            L2 = 0
            for params in net.parameters():
                # print(params.shape)
                L1 += torch.sum(torch.abs(params[0])).to(device)
                L2 += torch.sum(torch.pow(params[0], 2)).to(device)
            loss = loss_fn(out, y)
            loss = loss + gamma * alpha * L1 + (1 - gamma) * alpha * L2
            print(loss)
            out = torch.argmax(out, 1)
            train_loss += loss.item()
            train_acc += torch.sum(torch.eq(y.cpu(), out.detach().cpu()))
            optim.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()
        train_avgloss = train_loss / len(train_loader)
        train_avgacc = train_acc / len(train_loader)

        test_loss = 0
        test_acc = 0
        for i, (x, y) in enumerate(test_loader):
            # y = torch.zeros(y.size(0),10).scatter_(1,y.reshape(-1,1),1)
            x = x.to(device)
            y = y.to(device)
            out = net(x, epoch)
            loss = loss_fn(out, y)
            out = torch.argmax(out, 1)
            test_loss += loss.item()
            test_acc += torch.sum(torch.eq(y.cpu(), out.detach().cpu()))
        test_avgloss = test_loss / len(test_loader)
        test_avgacc = test_acc / len(test_loader)

        print("epoch:{},train_loss:{:.3f},test_loss:{:.3f}".format(epoch, train_avgloss, test_avgloss))
        print("epoch:{},train_acc:{:.3f},test_acc:{:.3f}".format(epoch, train_avgacc, test_avgacc))
        # writer.add_scalars("loss",{"train_loss":train_avgloss,"test_loss":test_avgloss},epoch)
        # writer.add_scalars("acc",{"train_acc":train_avgacc,"test_acc":test_avgacc},epoch)
        # torch.save(net.state_dict(), save_params)
        # torch.save(net, save_net)
        "tensorboard --logdir=路径"
