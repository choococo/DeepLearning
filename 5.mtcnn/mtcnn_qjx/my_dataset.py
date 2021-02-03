from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torchvision import transforms
from torch import nn
import random
import os
import cv2
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt


class My_dataset(Dataset):

    def __init__(self, img_path, is_train=True):
        super().__init__()

        if not is_train:
            self.img_path = r'G:/机器学习/01-代码/06深度学习/mine/06_MTCNN/data/VERIFY1' + img_path
        else:
            self.img_path = r'G:/机器学习/01-代码/06深度学习/mine/06_MTCNN/data/TRAIN' + img_path
        self.dataset = []

        '''读取文本文件'''
        self.dataset.extend(open(os.path.join(self.img_path, 'positive.txt'), encoding='utf-8').readlines())
        self.dataset.extend(open(os.path.join(self.img_path, 'part.txt'), encoding='utf-8').readlines())
        self.dataset.extend(open(os.path.join(self.img_path, 'negative.txt'), encoding='utf-8').readlines())


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        self.path, *a = self.dataset[index].strip().split(',')

        offset = torch.tensor([float(a[1]), float(a[2]), float(a[3]), float(a[3])])
        cond = torch.tensor([float(a[0])])

        img = Image.open(os.path.join(self.img_path, self.path))
        img = self.__transfor(img)
        # img=Image.open(os.path.join(self.img_path, self.path))
        # img.show()
        # exit()

        # img_data = torch.tensor((np.array(Image.open(os.path.join(self.img_path, self.path))) / 255. - 0.5) / 0.5,
        #                         dtype=torch.float32)
        # img = img_data.permute(2, 0, 1)
        # 归一化处理之后，图片的颜色会发生变化
        return img, cond, offset

    '''预处理'''

    def __transfor(self, x):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.486], std=[0.24, 0.225, 0.229])
        ])(x)


if __name__ == '__main__':
    img_path = '/48'
    my_data = My_dataset(img_path, False)
    print(len(my_data))
    data_loader = DataLoader(my_data, 6, True)
    print(data_loader.dataset)
    img, cond, offset = next(iter(data_loader))

    # exit()
    # print(cond)
    # print(img.shape)
    # for i ,(x,y,z) in enumerate(data_loader):
    #     #
    #     print(x)
    #     exit()
    #
    #     for j in range(6):
    #         plt.subplot(2,3,i+1)
    #         plt.imshow(x[j].permute(1,2,0))
    #         plt.show()
    #     print(x[0])
    #     exit()

    # exit()
