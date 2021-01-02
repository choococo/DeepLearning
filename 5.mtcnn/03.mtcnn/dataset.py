import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对于图像这个是有经验值的
])


class MyDataset(Dataset):

    def __init__(self, root, is_landmark=False):
        self.dataset = []
        self.root = root
        self.is_landmark = is_landmark
        self.dataset.extend(open(os.path.join(root, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(root, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(root, "part.txt")).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.is_landmark is False:
            img_filename, cls, off_x1, off_y1, off_x2, off_y2 = self.dataset[index].strip().split()[:6]
        else:
            img_filename, cls, off_x1, off_y1, off_x2, off_y2, off_px1, off_py1, off_px2, off_py2, off_px3, off_py3, off_px4, off_py4, off_px5, off_py5 = \
            self.dataset[index].strip().split()
        img_path = os.path.join(self.root, img_filename)
        cls = torch.tensor([int(cls)], dtype=torch.float32)
        offset = torch.tensor([float(off_x1), float(off_y1), float(off_x2), float(off_y2)], dtype=torch.float32)
        img_data = img_transform(Image.open(img_path))
        return img_data, cls, offset


if __name__ == '__main__':
    dataset = MyDataset(r"F:\2.Dataset\mtcnn_dataset\testing\test01\train\48")
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)

    data_loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4)
    for i, (img, cls, offset) in enumerate(data_loader):
        print(img.shape)
        print(cls.shape)
        print(cls)
        print(offset)
        exit()

"""
数据集结构：
    train:
        - 12
            -- positive
            -- part
            -- negative
               positive.txt
               part.txt
               negative.txt
        - 24 
            -- positive
            -- part
            -- negative
               positive.txt
               part.txt
               negative.txt
        - 48
            -- positive
            -- part
            -- negative
               positive.txt
               part.txt
               negative.txt
    val:
        ...
    test:
        ...
"""
