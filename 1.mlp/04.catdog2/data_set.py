import torch
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class CatDogDataset(Dataset):

    def __init__(self, root, flag=0):
        """
        初始化
        :param root: 根目录
        :param flag: 0：train, 1 : val, 2 : test
        """
        self.dataset = []
        choice = {0: "train", 1: "val", 2: "test"}
        sub_dir = choice[flag]
        for tag, img_filename in enumerate(os.listdir(os.path.join(root, sub_dir))):
            label = img_filename.split(".")[0]
            img_path = os.path.join(root, sub_dir, img_filename)
            self.dataset.append([img_path, int(label)]) # label需要是int类型

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img_data = img_transform(Image.open(img))
        return img_data, label


if __name__ == '__main__':
    datatset = CatDogDataset(r"F:\2.Dataset\cat_dog", 0)
    data_loader = DataLoader(datatset, batch_size=100, shuffle=False)
    for img, label in  data_loader:
        print(img)
        print(label)
        exit(0)