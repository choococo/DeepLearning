import torch, os, cv2
from torch.utils.data import Dataset
import numpy as np


class MNISTDataset(Dataset):

    def __init__(self, root, is_train=True):
        self.dataset = []
        sub_dir = "train" if is_train else "test"
        for tag in os.listdir(f"{root}/{sub_dir}"):  # tag是0-9的标签
            img_dir = f"{root}/{sub_dir}/{tag}"
            for img_filename in os.listdir(img_dir):  # 遍历0的文件夹
                img_path = f"{img_dir}/{img_filename}"
                self.dataset.append([img_path, tag])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index): # 每条数据的处理方式
        # 一条一条的取出数据
        data = self.dataset[index]
        # 打开图片
        img_data = cv2.imread(data[0], cv2.IMREAD_GRAYSCALE)  # 为了安全转为灰度图
        # 将数据变为一维的数据（数据展平）
        # print(img_data.shape)  # (28, 28)
        img_data = img_data.reshape(-1)
        # print(img_data.shape) # (784,)
        # 对数据做归一化(0-1)
        img_data = img_data / 255

        # 对标签做one-hot编码
        tag_one_hot = np.zeros(10) # 十分类
        tag_one_hot[int(data[1])] = 1

        return np.float32(img_data), np.float32(tag_one_hot)


if __name__ == '__main__':
    dataset = MNISTDataset(r"F:\2.Dataset\MNIST_DATASET")
    print(dataset[0][0].shape)
    print(dataset[0][1])

"""
dataset:
    train:
        0
            0.jpg
        1
        ...
        9
    test:
        0
        1
        ...
        9
"""
