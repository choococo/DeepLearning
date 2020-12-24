from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch


class MyDataset(Dataset):

    def __init__(self, main_dir, is_train=True):
        self.dataset = []
        data_filename = "TRAIN" if is_train else "TEST"
        # 循环获得样本数据文件夹下的训练集或测试集文件夹下的数字类别文件夹
        for i, cls_filename in enumerate(os.listdir(os.path.join(main_dir, data_filename))):
            # print(i)
            # print(os.listdir(os.path.join(main_dir)))
            # print(os.listdir(os.path.join(main_dir,data_filename)))
            # 循环获得每个类别文件夹下的数字图片
            for img_data in os.listdir(os.path.join(main_dir, data_filename, cls_filename)):
                # print(os.path.join(main_dir,data_filename,cls_filename,img_data))
                # 把每张人脸图片和对应的类别标签（类别文件夹的索引）放到一起
                self.dataset.append([os.path.join(main_dir, data_filename, cls_filename, img_data), i])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        image_data = self.image_preprocess(Image.open(data[0]))
        label_data = data[1]
        return image_data, label_data

    def image_preprocess(self, x):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ],
                                 std=[0.5, ])
        ])(x)


if __name__ == '__main__':
    data_path = r"E:\datasets\MNIST_IMG"
    dataset = MyDataset(data_path, True)

    dataloader = DataLoader(dataset, 100, shuffle=True, num_workers=0, drop_last=True)
    for data in dataloader:
        print(data[0].shape)
        print(data[1].shape)
