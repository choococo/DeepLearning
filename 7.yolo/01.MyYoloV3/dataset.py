import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import math
import cfg
from utils.tools import convert_to_416x416,center_to_left_and_right

LABEL_FILE_PATH = r"data/lable_class_3.txt"
IMG_BASE_DIR = r"G:\3class_2_group\process_img"
# IMG_BASE_DIR = r"F:\BaiduNetdiskDownload\VOCtrainval_11-May-2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages"

img_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def one_hot(cls_num, i):
    b = np.zeros(cls_num)
    b[i] = 1.
    return b


# class MyDataset(Dataset):
#
#     def __init__(self):
#         with open(LABEL_FILE_PATH, "r") as f:
#             self.dataset = f.readlines()
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         labels = {}  # {13:"..."}
#         line = self.dataset[index]
#         strs = line.strip().split()
#         img_name = strs[0]
#         _boxes = np.array([float(x) for x in strs[1:]])  # 这里需要转成array，不然下面不能使用array
#         boxes = np.split(_boxes, len(_boxes) // 5)
#         img_data = img_transform(Image.open(os.path.join(IMG_BASE_DIR, img_name)))
#         for feature_size, anchors in cfg.ANCHORS_GROUP.items():  # 得到对应尺寸下的建议框
#             labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))
#             # 对上面的标签boxes进行偏移量的计算
#             for box in boxes:
#                 cls, cx, cy, w, h = box
#                 # 中心点的偏移量，用整数代表中心点的格子，小数代表中心点的偏移量
#                 cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
#                 cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_HEIGHT)
#                 # 计算置信度、宽和高度的偏移量
#                 for i, anchor in enumerate(anchors):  # 一个建议框的宽和高
#                     anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]  # 13x13中的第一个建议框的面积
#
#                     p_w, p_h = w / anchor[0], h / anchor[1]
#                     p_area = w * h
#                     iou = min(p_area, anchor_area) / max(p_area, anchor_area)
#                     print(iou)
#                     labels[feature_size][int(cy_index), int(cx_index), i] = np.array([
#                         iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(cfg.CLASS_NUM, int(cls))
#                     ])
#         return labels[13], labels[26], labels[52], img_data

class MyDataset(Dataset):

    def __init__(self, flag=0):
        with open(LABEL_FILE_PATH) as f:
            if flag == 0:
                self.dataset = f.readlines()[:13700]
            elif flag == 1:
                self.dataset = f.readlines()[13700:15412]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}
        line = self.dataset[index]
        strs = line.split()
        # _, _, _, _img_data = convert_to_416x416(Image.open(os.path.join(IMG_BASE_DIR, strs[0])))
        _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))
        img_data = img_transform(_img_data)
        _boxes = np.array([float(x) for x in strs[1:]])

        boxes = np.split(_boxes, len(_boxes) // 5)
        # center_to_left_and_right(_img_data, boxes)

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))
            for box in boxes:
                cls, cx, cy, w, h = box
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)
                for i, anchor in enumerate(anchors):
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    # tw=np.log(w / anchor[0])
                    p_area = w * h
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)

                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h),
                         *one_hot(cfg.CLASS_NUM, int(cls))])  # 10,i
        return labels[13], labels[26], labels[52], img_data


if __name__ == '__main__':
    dataset = MyDataset()
    print(dataset[0][0].shape)
    data_loader = DataLoader(dataset, 2, shuffle=False)
    for i, (x, y, z, w) in enumerate(data_loader):
        print(x)
        # print()
