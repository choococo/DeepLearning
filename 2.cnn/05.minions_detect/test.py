import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyDataset
from net import Net
import os
import numpy as np
from torchvision import models, transforms
from sklearn.metrics import r2_score, explained_variance_score
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class MobileNet(nn.Module):  # 使用mobilenet网络进行学习
#     def __init__(self):
#         super(MobileNet, self).__init__()
#         self.mobile_net = models.mobilenet_v2()
#         self.mobile_net.classifier[1] = nn.Linear(1280, 5)
#
#     def forward(self, x):
#         out = self.mobile_net(x)
#         conf = torch.sigmoid(out[:, 4])
#         bbox = out[:, :4]
#         return conf, bbox


def evaluaion(net, test_loader, BATCH_SIZE):
    """
    测试评估指标
    :param net:
    :param test_loader:
    :param BATCH_SIZE:
    :return:
    """
    test_acc_list = []
    test_r2_list = []
    net.eval()

    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        conf_out, box_out = net(x)  # [N, 1] [N, 4]

        conf_label = y[:, 0]
        box_label = y[:, 1:]

        box_out = box_out.cpu().detach().numpy()
        box_label = box_label.cpu().detach().numpy()
        conf_out[conf_out >= 0.5] = 1.0
        conf_out[conf_out < 0.5] = 0.0
        val_accuracy = (torch.sum(torch.eq(conf_out, conf_label)) / BATCH_SIZE).item()

        r2 = explained_variance_score(box_label, box_out)
        test_r2_list.append(r2)
        test_acc_list.append(val_accuracy)
        # print(val_accuracy)

    test_r2_s = np.mean(test_r2_list)
    test_acc_s = np.mean(test_acc_list)
    print("测试集R2分数====>", test_r2_s)
    print("测试集Acc分数====>", test_acc_s)
    exit()


if __name__ == '__main__':

    BATCH_SIZE = 15 # 批次
    EPOCH = 5  # 需要训练的轮次
    root = r"F:\2.Dataset\Yellow\Minions"
    index = 0  # 权重索引，第一次训练为零

    train_dataset = MyDataset(root, flag=0)
    val_dataset = MyDataset(root, flag=1)
    test_dataset = MyDataset(root, flag=2)
    # print(len(val_dataset))
    # print(len(test_dataset))

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

    net = Net().cuda() # 放到cuda上

    net.load_state_dict(torch.load("./params/14.t")) # 加载参数
    # evaluaion(net, test_loader, BATCH_SIZE)

    net.eval() # 进入测试，在测试的时候一定要加上
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]

        x = img[None, ...].cuda()
        conf, box = net(x)
        box = box.reshape(-1)
        box = box.cpu().detach().numpy() * 224
        bbox = [int(x) for x in box]
        bbox = tuple(bbox)
        print(bbox)
        # print(*bbox)
        print(img.shape)
        '第一种方法反算图片数据'
        # img2 = img.cpu().detach() * 0.5 + 0.5
        # img2 = transforms.ToPILImage()(img2)
        # img2.show()
        # exit()
        '第二种方法反算图片数据'
        img = np.uint8((img.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255)

        image = Image.fromarray(img).convert("RGB") # 图片转换成RGB通道，一定要注意
        draw = ImageDraw.Draw(image) # 将图片放到画板上
        draw.rectangle(bbox, fill=None, outline="red", width=2) # 画框
        plt.imshow(img)
        plt.pause(0.1)
        plt.show()
        # image.show() # 显示图片
