import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(100, padding=6),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
other_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class MyDataset(Dataset):

    def __init__(self, root, flag=0):
        self.dataset = []
        self._flag = flag
        choice = {0: "train", 1: "val", 2: "test"}
        sub_dir = choice[self._flag]
        for img_filename in os.listdir(os.path.join(root, sub_dir)):
            label = img_filename.split(".")[0]
            img_path = os.path.join(root, sub_dir, img_filename)
            self.dataset.append([img_path, int(label)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self._flag == 0:
            img_data = img_transform(Image.open(img))
        else:
            img_data = other_transform(Image.open(img))
        return img_data, label


if __name__ == '__main__':
    dataset = MyDataset(r"F:\2.Dataset\cat_dog", 0)
    mask = dataset[0][0] > 0
    print(mask.shape)
    print(dataset[1][0].shape)
    print(dataset[0][1])
