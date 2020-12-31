import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from simpling import FaceDataset


class Trainer:
    def __init__(self, net, save_path, dataset_path):
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.net.to(self.device)

        self.cls_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters())
        # self.optimizer = optim.SGD(self.net.parameters(),lr=0.001)
        # self.optimizer = optim.SGD(self.net.parameters(),lr=0.001,momentum=0.5,weight_decay=0.0005)

        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))
        else:
            print("NO Param")

    def train(self, alpha):
        faceDataset = FaceDataset(self.dataset_path)
        dataloader = DataLoader(faceDataset, batch_size=20, shuffle=True, num_workers=4)
        loss = 0
        while True:
            for i, (img_data_, category_, offset_) in enumerate(dataloader):
                img_data_ = img_data_.to(self.device)
                category_ = category_.to(self.device)
                offset_ = offset_.to(self.device)

                _output_category, _output_offset = self.net(img_data_)
                output_category = _output_category.view(-1, 1)  # reshape
                output_offset = _output_offset.view(-1, 4)
                # output_landmark = _output_landmark.view(-1, 10)
                # 计算分类的损失
                # eq:等于，lt:小于，gt:大于，le:小于等于，ge:大于等于
                category_mask = torch.lt(category_, 2)  # 得到分类标签小于2的布尔值，a<2,[0,1,2]-->[1,1,0]
                category = torch.masked_select(category_, category_mask)  # 通过掩码，得到符合条件的置信度标签值
                output_category = torch.masked_select(output_category, category_mask)  # 得到符合条件的置信度输出值
                cls_loss = self.cls_loss_fn(output_category, category)  # 置信度损失

                # 计算bound的损失
                offset_mask = torch.gt(category_, 0)  # 得到分类标签大于0的布尔值,a>0,[0,1,2]-->[0,1,1]
                offset = torch.masked_select(offset_, offset_mask)  # 通过掩码，得到符合条件的偏移量标签值

                output_offset = torch.masked_select(output_offset, offset_mask)  # 得到符合条件的偏移量输出值
                offset_loss = self.offset_loss_fn(output_offset, offset)  # 偏移量损失

                loss = alpha * cls_loss + (1 - alpha) * offset_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cls_loss = cls_loss.cpu().item()
                offset_loss = offset_loss.cpu().item()
                loss = loss.cpu().item()
                print(" loss:", loss, " cls_loss:", cls_loss, " offset_loss", offset_loss)

            torch.save(self.net.state_dict(), self.save_path)
            print("save success")


if __name__ == '__main__':
    import nets

    net = nets.PNet()
    index = 0
    save_path = fr"./params_p/p_net_{index}.pth"
    img_dir = r"F:\2.Dataset\mtcnn_dataset\testing\test01/train/12"
    if not os.path.exists("./params_p"):
        os.makedirs("./params_p")
    trainer = Trainer(net, save_path, img_dir)
    trainer.train(0.7)
