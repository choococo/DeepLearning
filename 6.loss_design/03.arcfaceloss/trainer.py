import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import r2_score

from model import Net
import os
import numpy as np

if __name__ == '__main__':
    save_path = "./models/net_arcloss.pth"
    train_data = torchvision.datasets.MNIST(root=r"D:\Dataset\MNIST", download=True, train=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5, ], std=[0.5, ])
                                            ]))
    test_data = torchvision.datasets.MNIST(root=r"D:\Dataset\MNIST", download=True, train=False,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.5, ], std=[0.5,])
                                           ]))
    train_loader = data.DataLoader(dataset=train_data, shuffle=True, batch_size=100, num_workers=4)
    test_loader = data.DataLoader(dataset=test_data, shuffle=True, batch_size=100, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print("No Param")

    'CrossEntropyLoss()=torch.log(torch.softmax(None))+nn.NLLLoss()'
    'CrossEntropyLoss()=log_softmax() + NLLLoss() '
    'nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合'

    loss_fn = nn.NLLLoss()
    # optimizer = torch.optim.Adam(net.parameters())
    # optimizer = torch.optim.SGD(net.parameters(),lr=1e-3, momentum=0.9)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(net.parameters(),lr=1e-3, momentum=0.9, weight_decay=0.0005)

    epoch = 0
    while True:
        feat_loader = []
        label_loader = []
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            feature, output = net.forward(x)
            # print(feature.shape)  # torch.Size([100, 2])
            # print(output.shape)  # torch.Size([100, 10])

            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(y.shape)  # torch.Size([100])
            feat_loader.append(feature)
            label_loader.append(y)

            if i % 100 == 0:
                print("epoch:", epoch, "i:", i, "arcsoftmax_loss:", loss.item())

        feat = torch.cat(feat_loader, 0)
        labels = torch.cat(label_loader, 0)
        net.visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)

        torch.save(net.state_dict(), save_path)

        with torch.no_grad():
            feat_loader2 = []
            label_loader2 = []
            label_list = []
            output_list = []
            for i, (x, y) in enumerate(test_loader):  # 加验证集
                x = x.to(device)
                y = y.to(device)
                feature, output = net.forward(x)

                loss = loss_fn(output, y)

                feat_loader2.append(feature)  # 方便画图
                label_loader2.append(y)

                output = torch.argmax(output, 1)
                # print(output.shape)  # torch.Size([100])
                # print(y.shape)  # torch.Size([100])

                label_list.append(y.data.cpu().numpy().reshape(-1))  # 方便做r2_score
                output_list.append(output.data.cpu().numpy().reshape(-1))

                if i % 600 == 0:
                    print("epoch:", epoch, "i:", i, "validate_loss:", loss.item())

            feat2 = torch.cat(feat_loader2, 0)
            labels2 = torch.cat(label_loader2, 0)
            net.visualize2(feat2.data.cpu().numpy(), labels2.data.cpu().numpy(), epoch)

            # r2_score评估

            r2 = r2_score(label_list, output_list)
            print("验证集第{}轮, r2_score评估分类精度为：{}".format(epoch, r2))

        epoch += 1
        if epoch == 30:
            break
