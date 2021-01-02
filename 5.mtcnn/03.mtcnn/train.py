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
    def __init__(self, net, save_params, sub_dir, index, face_size, is_landmark=False):
        self.net = net
        self.save_params = save_params
        self.choice = {0: "train", 1: "val"}
        self.dataset_path = sub_dir
        self.index = index
        self.face_size = face_size
        self.is_landmark = is_landmark

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
        print(self.dataset_path)
        batch_size = 512
        summarywriter = SummaryWriter(f"./logs{self.face_size}")

        train_dataset = MyDataset(f"{self.dataset_path}/{self.choice[0]}/{self.face_size}", self.is_landmark)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = MyDataset(f"{self.dataset_path}/{self.choice[1]}/{self.face_size}", self.is_landmark)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        print(len(train_dataset))
        train_loss_list = []
        train_acc_list = []
        train_r2_list = []

        val_loss_list = []
        val_acc_list = []
        val_r2_list = []

        r2_max = 0
        average_num = 5
        start = 0
        epoch = 0
        # for epoch in range(EPOCH):
        while True:
            train_loss_list_inter = []
            train_acc_list_inter = []
            train_r2_list_inter = []
            for i, (img_data_, cls_, offset_) in enumerate(train_loader):
                img_data_label = img_data_.to(self.device)
                cls_label = cls_.to(self.device)
                offset_label = offset_.to(self.device)

                # 网络输出
                output_cls, output_offset = self.net(img_data_label)
                output_cls = output_cls.reshape(-1, 1)
                output_offset = output_offset.reshape(-1, 4)

                # 掩码计算
                mask_cls = torch.lt(cls_label, 2)
                cls = torch.masked_select(cls_label, mask_cls)  # 从label中找
                out_cls = torch.masked_select(output_cls, mask_cls)  # 从网络输出中找
                # print(out_cls.shape, cls.shape)  # torch.Size([9]) torch.Size([9]) 将数据展平
                cls_loss = self.cls_loss_fn(out_cls, cls)  # 计算损失

                # 拿到对应的掩码值
                cls = cls_label[mask_cls]
                mask_offset = torch.gt(cls_label, 0)
                # mask_offset = cls_label[:, 0] > 0
                offset = torch.masked_select(offset_label, mask_offset)  # 从label中找
                out_offset = torch.masked_select(output_offset, mask_offset)  # 从网络输出中找
                offset_loss = self.offset_loss_fn(out_offset, offset)

                # 计算总损失
                # p_net中更加注重置信度，因此置信度前面的系数加大
                loss = alpha * cls_loss + (1 - alpha) * offset_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cls_loss = cls_loss.cpu().item()
                offset_loss = offset_loss.cpu().item()
                loss = loss.cpu().item()
                out_cls[out_cls > 0.5] = 1
                out_cls[out_cls <= 0.5] = 0

                acc = accuracy_score(out_cls.cpu().detach(), cls.cpu().detach())  # 计算精度
                r2 = r2_score(offset.cpu().detach(), out_offset.cpu().detach())  # 计算r2分数

                train_acc_list_inter.append(acc)
                train_loss_list_inter.append(loss)
                train_r2_list_inter.append(r2)

            train_acc_list.append(np.mean(train_acc_list_inter))
            train_loss_list.append(np.mean(train_loss_list_inter))
            train_r2_list.append(np.mean(train_r2_list_inter))

            print(
                f"{epoch + self.index + 1} | train_acc：{np.mean(train_acc_list_inter)} | train_loss: {np.mean(train_loss_list_inter)} | r2_score: {np.mean(train_r2_list_inter)}")

            val_acc_list_inter = []
            val_loss_list_inter = []
            val_r2_list_inter = []
            self.net.eval()
            for i, (img_data_, category_, offset_) in enumerate(val_loader):
                img_data_label = img_data_.to(self.device)
                cls_label = category_.to(self.device)
                offset_label = offset_.to(self.device)

                # 网络输出
                output_cls, output_offset = self.net(img_data_label)
                output_cls = output_cls.reshape(-1, 1)
                output_offset = output_offset.reshape(-1, 4)

                # 掩码计算
                mask_cls = torch.lt(cls_label, 2)
                cls = torch.masked_select(cls_label, mask_cls)  # 从label中找
                out_cls = torch.masked_select(output_cls, mask_cls)  # 从网络输出中找
                # print(out_cls.shape, cls.shape)  # torch.Size([9]) torch.Size([9]) 将数据展平
                cls_loss = self.cls_loss_fn(out_cls, cls)  # 计算损失

                # 拿到对应的掩码值
                cls = cls_label[mask_cls]
                mask_offset = torch.gt(cls_label, 0)
                # mask_offset = cls_label[:, 0] > 0
                offset = torch.masked_select(offset_label, mask_offset)  # 从label中找
                out_offset = torch.masked_select(output_offset, mask_offset)  # 从网络输出中找
                offset_loss = self.offset_loss_fn(out_offset, offset)

                # 计算总损失
                # p_net中更加注重置信度，因此置信度前面的系数加大
                loss = alpha * cls_loss + (1 - alpha) * offset_loss

                cls_loss = cls_loss.cpu().item()
                offset_loss = offset_loss.cpu().item()
                loss = loss.cpu().item()
                out_cls[out_cls > 0.5] = 1
                out_cls[out_cls <= 0.5] = 0

                acc = accuracy_score(out_cls.cpu().detach(), cls.cpu().detach())  # 计算精度
                r2 = r2_score(offset.cpu().detach(), out_offset.cpu().detach())  # 计算r2分数

                val_acc_list_inter.append(acc)
                val_loss_list_inter.append(loss)
                val_r2_list_inter.append(r2)

            val_acc_list.append(np.mean(val_acc_list_inter))
            val_loss_list.append(np.mean(val_loss_list_inter))
            val_r2_list.append(np.mean(val_r2_list_inter))

            val_r2_s = np.mean(val_r2_list_inter)

            val_acc_list.append(np.mean(val_acc_list_inter))
            val_loss_list.append(np.mean(val_loss_list_inter))
            val_r2_list.append(np.mean(val_r2_list_inter))
            print(
                f"{epoch + self.index + 1} | val_acc  ：{np.mean(val_acc_list_inter)} "
                f"| val_loss  : {np.mean(val_loss_list_inter)} | r2_score : {np.mean(val_r2_list_inter)}")
            train_avg_loss = train_loss_list[epoch]
            val_avg_loss = val_loss_list[epoch]
            train_avg_score = train_acc_list[epoch]
            val_avg_score = val_acc_list[epoch]
            train_r2_score = train_r2_list[epoch]
            val_r2_score = val_r2_list[epoch]

            summarywriter.add_scalars("loss", {"train_loss": train_avg_loss, "val_loss": val_avg_loss}, epoch)
            summarywriter.add_scalars("acc", {"train_score": train_avg_score, "val_score": val_avg_score}, epoch)
            summarywriter.add_scalars("R2", {"train_r2_score": train_r2_score, "val_r2_score": val_r2_score}, epoch)

            if val_r2_s > r2_max:
                r2_max = val_r2_s
                # ./params_p/p_net_{index}.pth
                if self.face_size == 12:
                    torch.save(self.net.state_dict(), f"params_p/p_net_{epoch + self.index + 1}.pth")
                elif self.face_size == 24:
                    torch.save(self.net.state_dict(), f"params_r/r_net_{epoch + self.index + 1}.pth")
                else:
                    torch.save(self.net.state_dict(), f"params_o/o_net_{epoch + self.index + 1}.pth")
                print("params save success.")
            print("====>", val_r2_s)

            # 加入过拟合自动判断，让程序过拟合自动停止
            r2_length = len(val_r2_list)
            if (len(val_r2_list) - start) % 10 == 0:
                last_avg_acc = np.mean(val_r2_list[start:r2_length - average_num])
                moment_avg_acc = np.mean(val_r2_list[r2_length - average_num:r2_length])
                if last_avg_acc > moment_avg_acc:
                    print(max(val_r2_list))
                    exit()
                start += average_num
            epoch += 1
            summarywriter.close()


if __name__ == '__main__':
    pass
