import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from dataset import MyDataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score, accuracy_score
import numpy as np


# from cfg import *

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对于图像这个是有经验值的
# ])


class Trainer:
    # dataset_path=F:\2.Dataset\mtcnn_dataset\testing\test01/{train/val}/{face_size}
    def __init__(self, net, save_params, sub_dir, face_size):
        self.net = net
        self.save_params = save_params
        self.choice = {0: "train", 1: "val"}
        self.dataset_path = sub_dir
        self.face_size = face_size

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.net.to(self.device)

        self.cls_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.net.parameters())
        # self.optimizer = torch.optim.SGD(self.net.parameters(),lr=0.001)
        # self.optimizer = torch.optim.SGD(self.net.parameters(),lr=0.001,momentum=0.5,weight_decay=0.0005)

        if os.path.exists(self.save_params):
            self.net.load_state_dict(torch.load(self.save_params))
            print("params loaded success...")
        print("No params.")

    def train(self, alpha):
        BATCH_SIZE = 10
        EPOCH = 50
        index = 0

        train_dataset = MyDataset(f"{self.dataset_path}/{self.choice[0]}/{self.face_size}")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = MyDataset(f"{self.dataset_path}/{self.choice[1]}/{self.face_size}")
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

        while True:
            for i, (img_data_, cls_, offset_) in enumerate(train_loader):
                img_data_label = img_data_.to(self.device)
                cls_label = cls_.to(self.device)
                offset_label = offset_.to(self.device)

                # 网络输出
                output_cls, output_offset = self.net(img_data_label)  # [N, C, H, W]
                output_cls = output_cls.reshape(-1, 1)
                output_offset = output_offset.reshape(-1, 4)

                # 掩码计算
                mask_cls = torch.lt(cls_label, 2)  # 正样本和负样本训练置信度的损失
                cls = torch.masked_select(cls_label, mask_cls)  # 从label中找
                out_cls = torch.masked_select(output_cls, mask_cls)  # 从网络输出中找
                # print(out_cls.shape, cls.shape)  # torch.Size([9]) torch.Size([9]) 将数据展平
                cls_loss = self.cls_loss_fn(out_cls, cls)  # 计算损失

                # 拿到对应的掩码值

                mask_offset = torch.gt(cls_label, 0)  # 正样本和部分样本训练回归的损失
                # mask_offset = cls_label[:, 0] > 0
                offset = torch.masked_select(offset_label, mask_offset)  # 从label中找

                out_offset = torch.masked_select(output_offset, mask_offset)  # 从网络输出中找
                offset_loss = self.offset_loss_fn(out_offset, offset)  # 计算损失

                # 计算总损失
                # p_net中更加注重置信度，因此置信度前面的系数加大
                loss = alpha * cls_loss + (1 - alpha) * offset_loss
                self.optimizer.zero_grad()  # 梯度清空
                loss.backward()  # 反向传播
                self.optimizer.step()  # 走一步

                cls_loss = cls_loss.cpu().item()
                offset_loss = offset_loss.cpu().item()
                loss = loss.cpu().item()

                out_cls[out_cls > 0.5] = 1
                out_cls[out_cls <= 0.5] = 0

                acc = accuracy_score(out_cls.cpu().detach(), cls.cpu().detach())  # 计算精度
                print(acc)
                r2 = r2_score(offset.cpu().detach(), out_offset.cpu().detach())  # 计算r2分数
                print(r2)
                exit()


if __name__ == '__main__':
    import net

    net = net.PNet()
    net.parameters()
    index = 0
    save_path = fr"./params_p/p_net_{index}.pth"
    img_dir = r"F:\2.Dataset\mtcnn_dataset\testing\test01"
    if not os.path.exists("./params_p"):
        os.makedirs("./params_p")
    trainer = Trainer(net, save_path, img_dir, 12)
    trainer.train(0.7)
