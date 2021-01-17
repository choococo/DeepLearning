import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim import lr_scheduler
from model import Net
import os
import numpy as np

if __name__ == '__main__':
    save_path = "params/net_softmax_loss.pt"
    train_data = datasets.MNIST(root=r"D:\Dataset\MNIST", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, ], std=[0.5, ])
    ]))
    train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=100, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Net().to(device)
    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print("No Params")

    'CrossEntropyLoss()=torch.log(torch.softmax(None))+nn.NLLLoss()'
    'CrossEntropyLoss()=log_softmax() + NLLLoss() '
    'nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合'

    loss_func = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters())
    # optimzer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    # optimzer = torch.optim.SGD(net.parameters(),lr=1e-3)
    # optimzer = torch.optim.SGD(net.parameters(),lr=1e-3, momentum=0.9, weight_decay=0.0005)

    epoch = 0
    while True:
        feat_loader = []
        label_loader = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            feature, output = net(x)
            loss = loss_func(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            feat_loader.append(feature)
            label_loader.append(y)

            if i % 10 == 0:
                print("epoch:", epoch, "i:", i, "total:", loss.item())
        feat = torch.cat(feat_loader, 0)
        labels = torch.cat(label_loader, 0)
        '---------------'
        # print(feat.shape)#feat.shape=[60000,2]
        # print(labels.shape)#feat.shape=[60000]
        '-------------------'
        net.visualize(feat.detach().cpu().numpy(), labels.detach().cpu().numpy(), epoch)
        epoch += 1
        torch.save(net.state_dict(), save_path)



