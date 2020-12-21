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


def test(net, test_loader, BATCH_SIZE):
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
    # net.eval()
    # tes_loss_list = []
    # tes_r2_list = []
    # tes_acc_list = []
    # for tes_data, tes_label in test_loader:
    #     tes_data, tes_label = tes_data.to(DEVICE), tes_label.to(DEVICE)
    #
    #     tes_out_confidence, tes_out_location = net(tes_data)
    #     tes_label_confidence,tes_label_location = tes_label[:, 0], tes_label[:, 1:]
    #
    #     tes_out_location = tes_out_location.detach().cpu().numpy()
    #     tes_label_location = tes_label_location.detach().cpu().numpy()
    #     tes_r2 = r2_score(tes_out_location, tes_label_location)
    #     tes_r2_list.append(tes_r2)
    #
    #     tes_out_confidence[tes_out_confidence >= 0.5] = 1.0
    #     tes_out_confidence[tes_out_confidence < 0.5] = 0.0
    #     tes_acc = (torch.sum(torch.eq(tes_out_confidence, tes_label_confidence)) / BATCH_SIZE).item()
    #     tes_acc_list.append(tes_acc)
    #     print(tes_acc)
    #
    # # tes_loss_total.append(np.mean(tes_loss_list))
    # # tes_r2_total.append(np.mean(tes_r2_list))
    # # tes_acc_total.append(np.mean(tes_acc_list))
    # print(f"tes_loss：{np.mean(tes_loss_list)}  |  "
    #       f"tes_R2_score：{np.mean(tes_r2_list)} | "
    #       f"tes_acc：{np.mean(tes_acc_list)}")
    exit()


if __name__ == '__main__':

    BATCH_SIZE = 15
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

    net = Net().cuda()

    net.load_state_dict(torch.load("./params/14.t"))
    # test(net, test_loader, BATCH_SIZE)

    net.eval()
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
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
        print(img.shape)
        # img2 = img.cpu().detach() * 0.5 + 0.5
        # img2 = transforms.ToPILImage()(img2)
        # img2.show()
        # exit()
        img = np.uint8((img.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255)
        image = Image.fromarray(img).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox, fill=None, outline="red", width=2)
        image.show()
