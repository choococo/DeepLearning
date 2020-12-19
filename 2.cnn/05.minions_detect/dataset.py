import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(100, padding=6),
    transforms.Normalize(mean=[0.5, ], std=[0.5, ])
])
other_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, ], std=[0.5, ])
])


class MyDataset(Dataset):

    def __init__(self, root, flag=0):
        self.dataset = []
        self._flag = flag
        choice = {0: "train", 1: "val", 2: "test"}
        sub_title = choice[self._flag]
        label_txt_dir = f"{root}/{sub_title}_label.txt"
        # 读取txt文本
        with open(label_txt_dir, "r") as f:
            for line in f.readlines():
                line = line.strip().split(" ")
                img_name = fr"{root}/{sub_title}/{line[0]}"
                label = [int(x) for x in line[1:]]  # 将标签中的值变成int
                self.dataset.append([img_name, label])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        label = torch.Tensor(np.array(label, dtype=np.float32))
        label[1:] = label[1:] / 224
        if self._flag == 0:
            # 这里无论什么时候，一定要转三个通道
            img_data = img_transform(Image.open(img).convert("RGB"))
        else:
            img_data = img_transform(Image.open(img).convert("RGB"))
        # label[conf x1 y1 x2 y2]
        return img_data, label


if __name__ == '__main__':
    # dataset = MyDataset(r"F:\2.Dataset\Yellow\Minions", 0)
    dataset = MyDataset(r"F:\2.Dataset\Yellow\Minions2", 2)
    # mask = dataset[0][0] > 0
    # print(mask.shape)
    # print(dataset[1][0].shape)
    # print(dataset[0][1])
    from torch.utils.data import DataLoader

    print(len(dataset))

    data_loader = DataLoader(dataset, batch_size=30, shuffle=False)
    for x, y in data_loader:
        print(y)
        print(y[0])
        exit()
