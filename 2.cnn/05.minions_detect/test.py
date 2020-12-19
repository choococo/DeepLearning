import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyDataset
from net import Net
import os
import numpy as np
from torchvision import models
from sklearn.metrics import r2_score
from PIL import Image,ImageDraw
import cv2
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MobileNet(nn.Module):  # 使用mobilenet网络进行学习
    def __init__(self):
        super(MobileNet, self).__init__()
        self.mobile_net = models.mobilenet_v2()
        self.mobile_net.classifier[1] = nn.Linear(1280, 5)

    def forward(self, x):
        out = self.mobile_net(x)
        conf = torch.sigmoid(out[:, 4])
        bbox = out[:, :4]
        return conf, bbox


if __name__ == '__main__':

    BATCH_SIZE = 15
    EPOCH = 35  # 需要训练的轮次
    root = r"F:\2.Dataset\Yellow\Minions"
    index = 0  # 权重索引，第一次训练为零

    train_dataset = MyDataset(root, flag=0)
    val_dataset = MyDataset(root, flag=1)
    test_dataset = MyDataset(root, flag=2)
    # print(len(val_dataset))
    # print(len(test_dataset))

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, 10, shuffle=True)

    net = Net().cuda()

    net.load_state_dict(torch.load("./params/19.t"))
    net.eval()
    for i in range(len(val_dataset)):
        img, label = val_dataset[i]
        # print(label)
        # exit()
        x = img[None, ...].cuda()
        conf, box = net(x)
        box = box.reshape(-1)
        box = box.cpu().detach().numpy() * 224
        bbox = [int(x) for x in box]
        bbox = tuple(bbox)
        print(bbox)
        # print(*bbox)

        img = np.uint8((img.permute(1, 2, 0).numpy()*0.5+0.5) * 255)
        image = Image.fromarray(img).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox, fill=None, outline="red", width=2)
        image.show()
    # 0.png 0 0 0 0 0
    # 0_bg.png 1 55 54 136 145
    image = Image.open(r"F:\2.Dataset\Yellow\Minions\train/0.png")
    draw  = ImageDraw.Draw(image)
    draw.rectangle(())


