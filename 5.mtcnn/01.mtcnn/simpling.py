from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from PIL import Image


class FaceDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")
        img_path = os.path.join(self.path, strs[0])
        cls = torch.tensor([int(strs[1])], dtype=torch.float32)
        offset = torch.tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])], dtype=torch.float32)
        img_data = torch.tensor((np.array(Image.open(img_path)) / 255. - 0.5) / 0.5, dtype=torch.float32)
        # print(img_data.shape)
        img_data = img_data.permute(2, 0, 1)

        return img_data, cls, offset


if __name__ == '__main__':
    dataset = FaceDataset(r"F:\2.Dataset\mtcnn_dataset\testing\test01/train/12")
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    dataloader = DataLoader(dataset, 5, shuffle=True, num_workers=4)
    for i, (img, cls, offset) in enumerate(dataloader):
        print(img.shape)
        print(cls.shape)
        print(cls)
        print(offset.shape)
