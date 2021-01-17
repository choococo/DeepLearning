import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim import lr_scheduler
from model import Net
import os
import numpy as np
from center_lossV2 import CenterLoss

if __name__ == '__main__':
    save_path = "params/net_center_loss.pt"
    train_data = datasets.MNIST(root=r"D:\Dataset\MNIST", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, ], std=[0.5, ])
    ]))
    train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=100, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Net().to(device)
    if os.path.exists(save_path) and os.path.exists("./params/center_net.pt"):
        net.load_state_dict(torch.load(save_path))
    else:
        print("No Params")

    'CrossEntropyLoss()=torch.log(torch.softmax(None))+nn.NLLLoss()'
    'CrossEntropyLoss()=log_softmax() + NLLLoss() '
    'nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合'
    # loss_func_class = nn.CrossEntropyLoss()
    loss_func_class = nn.NLLLoss()
    loss_func_center = CenterLoss(lambdas=2, feature_num=2, class_num=10).to(device)
    # optimzer = torch.optim.SGD(net.parameters(),lr=1e-3, momentum=0.9, weight_decay=0.0005)
    optimzer_softmax = torch.optim.Adam(net.parameters())
    "定义专门优化centerloss里的nn.Parameter权重的优化器"
    optimzer_cenetr = torch.optim.SGD(loss_func_center.parameters(), lr=0.5)
    scheduler = lr_scheduler.StepLR(optimzer_softmax, 10, gamma=0.9)
    epoch = 0
    while True:
        feat_loader = []
        label_loader = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            feature, output = net(x)
            loss_class = loss_func_class(output, y)
            loss_center = loss_func_center(feature, y)

            loss = loss_class + loss_center
            optimzer_softmax.zero_grad()
            optimzer_cenetr.zero_grad()
            loss_class.backward(retain_graph=True)
            loss_center.backward()
            optimzer_softmax.step()
            optimzer_cenetr.step()

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
        torch.save(loss_func_center.state_dict(), "./params/center_net.pt")
        "更新学习率，epoch给空值，给指定值的话，就代表学习率一直是当前值，无法更新"
        scheduler.step(epoch=None)
        lr = scheduler.get_last_lr()
