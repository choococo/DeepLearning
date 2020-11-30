from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os

train_dataset = datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST("./data", train=False, transform=transforms.ToTensor(), download=False)

print(train_dataset.data.shape)
print(train_dataset.targets.shape)

save_path = r"F:\2.Dataset\MNIST_DATASET1"
train_path = os.path.join(save_path, "train")
test_path = os.path.join(save_path, "test")

try:
    for dir in [save_path, train_path, test_path]:
        if not os.path.exists(dir):
            os.mkdir(dir)
    for i in range(10):
        if not os.path.exists(os.path.join(train_path, str(i))):
            os.mkdir(os.path.join(train_path, str(i)))
        if not os.path.exists(os.path.join(test_path, str(i))):
            os.mkdir(os.path.join(test_path, str(i)))
except Exception as e:
    print(e)

for dataset in [train_dataset]:
    _dict = {j: 0 for j in range(10)}
    print(_dict)
    for i, (image, label) in enumerate(dataset):
        _dict[label] += 1
        img = transforms.ToPILImage()(image)
        # plt.imshow(img, cmap="gray")
        # plt.savefig(f"{train_path}/{label}/{_dict[label]}.png")
        img.save(f"{train_path}/{label}/{_dict[label]}.png")
    print(sum(_dict.values()))

for dataset in [test_dataset]:
    _dict = {j: 0 for j in range(10)}
    print(_dict)
    for i, (image, label) in enumerate(dataset):
        _dict[label] += 1
        img = transforms.ToPILImage()(image)
        img.save(f"{test_path}/{label}/{_dict[label]}.png")
    print(sum(_dict.values()))