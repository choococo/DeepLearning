from torchsummary import summary
from net import CatDogNet
import torch


if __name__ == '__main__':
    model = CatDogNet().cuda()
    summary(model, input_size=(3, 100, 100))
    


