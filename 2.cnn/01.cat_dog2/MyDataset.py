import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(100, padding=6),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

other_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class CatDogDataset(Dataset):

    def __init__(self, root, flag=0):
        self.dataset = []
        self.flag = flag
        choice = {0: "train", 1: "val", 2: "test"}
        sub_dir = choice[self.flag]
        for img_filename in os.listdir(os.path.join(root, sub_dir)):
            label = img_filename.split(".")[0]
            img_path = os.path.join(root, sub_dir, img_filename)
            self.dataset.append([img_path, int(label)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = Image.open(image)
        if self.flag == 0:
            image = img_transform(image)
        else:
            image = other_transform(image)
        return image, label


if __name__ == '__main__':
    dataset = CatDogDataset(r"F:\2.Dataset\cat_dog", 0)
    mask = dataset[0][0] > 0
    print(mask.shape)
    print(dataset[1][0].shape)
    print(dataset[0][1])
