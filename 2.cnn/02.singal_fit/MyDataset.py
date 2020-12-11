import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os


class WaveformDataset(Dataset):

    def __init__(self, is_train=True):
        self.dataset = []
        self.train_data = self._train_data()[0]
        self.based_data = self._train_data()[1]
        for i in range(len(self.train_data) - 9):
            data = torch.FloatTensor(self.train_data[i:i + 9])
            tag = torch.FloatTensor(self.train_data[i + 9:i + 10])
            self.dataset.append([data, tag])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label

    def _train_data(self):
        np.random.seed(0)
        x = np.random.uniform(0, 10, (10,))
        np.random.seed()
        y_list = []
        # 这里面目前不太好，
        for j in range(30):
            for i in x:
                y = i + np.random.uniform(-0.5, 0.5)
                y_list.append(y)
        return y_list, x


if __name__ == '__main__':
    # for i in range(2):
    dataset = WaveformDataset()
    base_data = dataset.based_data
    base_data_list = []
    for i in range(30):
        base_data_list.extend(base_data)
    print(len(base_data_list))
    train = dataset.train_data
    print(len(train))

    import matplotlib.pyplot as plt

    plt.plot([i for i in range(291)], base_data_list[9:])
    plt.plot([i for i in range(291)], train[9:])
    plt.show()

