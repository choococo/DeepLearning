import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from darknet53 import MainNet
from dataset import MyDataset
from torch.utils.tensorboard import SummaryWriter
import cfg
from sklearn.metrics import r2_score, accuracy_score


def loss_fn(output, target, alpha=0.5):
    conf_loss_fn = nn.BCEWithLogitsLoss()  # 置信度使用带sigmoid的二值交叉熵进行计算
    bound_loss_fn = nn.MSELoss()  # 中心点和坐标偏移量的损失, 可以拆成下面两个损失
    # center_loss_fn = nn.BCEWithLogitsLoss()  # 中心点的损失
    # wh_loss_fn = nn.MSELoss()  # 宽和高的损失
    cls_loss_fn = nn.CrossEntropyLoss()  # 分类使用了交叉熵，如果类别之间是有交集的，使用BCELoss，sigmoid没有排他性
    # output.shape=[N, C, H, W] - >[N, H, W, C]
    output = output.permute(0, 2, 3, 1)
    # output.shape=[N, H, W, 3, 5 + cls_num]
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
    output = output.cpu().double()
    # target.shape=[13, 13, 3, 5 + cls_num]
    mask_obj = target[..., 0] > 0  # 取出置信度  四维的
    output_obj = output[mask_obj]  # 用五维取索引思维维，得到二维
    target_obj = target[mask_obj]
    loss_obj_conf = conf_loss_fn(output_obj[:, 0], target_obj[:, 0])
    # loss_obj_center = center_loss_fn(output_obj[:, 1:3], target_obj[:, 1:3])
    # loss_obj_wh = wh_loss_fn(output_obj[:, 3:5], target_obj[:, 3:5])
    loss_obj_bound = bound_loss_fn(output_obj[:, 1:5], target_obj[:, 1:5])
    r2_obj_bound = r2_score(target_obj[:, 1:5].detach().numpy(), output_obj[:, 1:5].detach().numpy())

    loss_obj_cls = cls_loss_fn(output_obj[:, 5:], torch.argmax(target_obj[:, 5:], dim=1))
    loss_obj = loss_obj_conf + loss_obj_bound + loss_obj_cls

    mask_no_obj = target[..., 0] == 0
    output_no_obj = output[mask_no_obj]
    target_no_obj = target[mask_no_obj]
    loss_no_obj = conf_loss_fn(output_no_obj[:, 0], target_no_obj[:, 0])
    # print("====>", loss_obj_conf.item(), loss_obj_bound.item(), loss_obj_cls.item())
    # 一般会将alpha的值给的大一些，让优化器更加关注有物体的，因为负样本占比很大
    total_loss = alpha * loss_obj + (1 - alpha) * loss_no_obj

    return total_loss, r2_obj_bound


def r2_acc_score(output):
    output = output.permute(0, 2, 3, 1)
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
    output = output.cpu().double()
    return output


if __name__ == '__main__':
    index = 0
    save_path = f"checkpoints_voc/yolo_voc_{index}.pt"
    # save_path = fr"F:\workspace\7.YOLO\01.MyYolov3\checkpoints_voc\yolo_voc_1_1899.pt"
    train_dataset = MyDataset(0)
    val_dataset = MyDataset(1)
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MainNet(cfg.CLASS_NUM).to(device)

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
        print("loaded params success...")
    else:
        print("No params.")

    optimizer = torch.optim.Adam(net.parameters())
    # optimizer = torch.optim.SGD(net.parameters(), lr=3e-4, momentum=0.0001, weight_decay=0.001)

    epoch = 0
    alpha = 0.5
    train_loss_list = []
    val_loss_list = []
    last_loss = 10.0
    while True:
        net.train()
        train_loss_list_inter = []
        for i, (target_13, target_26, target_52, img_data) in enumerate(train_loader):
            img_data = img_data.to(device)
            output_13, output_26, output_52 = net(img_data)

            loss_13, train_r2_13 = loss_fn(output_13, target_13, alpha)
            loss_26, train_r2_26 = loss_fn(output_26, target_26, alpha)
            loss_52, train_r2_52 = loss_fn(output_52, target_52, alpha)

            loss = loss_13 + loss_26 + loss_52

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # output_13 [N , 13, 13, 3, 5+cls_num]
            # conf, cx,cy, w, h, cls

            train_r2 = train_r2_13 + train_r2_26 + train_r2_52
            train_loss_list_inter.append(loss.item())
            if i % 5 == 0:
                print(f"{epoch}-{i} | train_loss:{loss.item()} | train_r2:{train_r2}")
        train_loss_list.append(np.mean(train_loss_list_inter))
        avg_loss = np.mean(train_loss_list_inter)
        # summary_writer.add_scalar("train_avg_loss", train_loss_list[epoch], (epoch + index + 1))
        print(f"{[epoch + index + 1]} train_avg_loss: {train_loss_list[epoch]}")  # 4.5274751075697415

        net.eval()
        val_loss_list_inter = []
        for i, (val_target_13, val_target_26, val_target_52, val_img_data) in enumerate(val_loader):
            val_img_data = val_img_data.to(device)
            val_output_13, val_output_26, val_output_52 = net(val_img_data)

            val_loss_13, val_r2_13 = loss_fn(val_output_13, val_target_13, alpha)
            val_loss_26, val_r2_26 = loss_fn(val_output_26, val_target_26, alpha)
            val_loss_52, val_r2_52 = loss_fn(val_output_52, val_target_52, alpha)

            val_loss = val_loss_13 + val_loss_26 + val_loss_52

            val_r2 = val_r2_13 + val_r2_26 + val_r2_52
            val_loss_list_inter.append(val_loss.item())
            if i % 5 == 0:
                print(f"{epoch}-{i} | val__loss:{val_loss.item()} | val__r2:{val_r2}")
            if (i + 1) % 100 == 0:
                torch.save(net.state_dict(), fr"checkpoints_voc/yolo_voc_{epoch + index + 1}_{i}.pt")
        val_loss_list.append(np.mean(val_loss_list_inter))
        print(f"{[epoch + index + 1]} val__avg_loss: {val_loss_list[epoch]}")  # 4.5274751075697415
        print()

        if avg_loss < last_loss:
            last_loss = avg_loss
            torch.save(net.state_dict(), fr"checkpoints_voc/yolo_voc_{epoch + index + 1}.pt")
            print(f"[PARAMS] save {epoch + index + 1}.pt success")
        epoch += 1
        # summary_writer.close()
