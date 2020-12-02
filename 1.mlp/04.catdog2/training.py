import torch
import torch.nn as nn
import numpy as np
from data_set import CatDogDataset
from torch.utils.data import DataLoader
from net import CatDogNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train:

    def __init__(self, root, batch_size=100):
        self.train_dataset = CatDogDataset(root, 0)
        self.val_dataset = CatDogDataset(root, 1)
        self.test_dataset = CatDogDataset(root, 2)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

        self.net = CatDogNet().to(DEVICE)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def __call__(self):
        pass

    def training(self):
        pass

    def validation(self):
        pass

    def test(self):
        pass


if __name__ == '__main__':
    BATCH_SIZE = 100
    train = Train(r"F:\2.Dataset\cat_dog")
    train()
