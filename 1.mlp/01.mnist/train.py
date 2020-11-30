import torch
from data import MNISTDataset
from net import Net
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

"""
直接安装 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard
"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# writer = SummaryWriter()
class Train:

    def __init__(self, root):
        # 使用收集器
        # self.sunnaryWriter = SummaryWriter("logs")
        # 加载训练数据
        self.train_dataset = MNISTDataset(root, is_train=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=100, shuffle=True)

        # 加载测试集
        self.test_dataset = MNISTDataset(root, is_train=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=100, shuffle=True)

        # 创建模型
        self.net = Net()
        # 将模型加载到设备上
        self.net.to(DEVICE)

        # 创建优化器, 使用Adam优化器自动调整学习率
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def __call__(self):
        for epoch in range(100000):
            sum_loss = 0
            for i, (image, tag) in enumerate(self.train_loader):
                image, tag = image.to(DEVICE), tag.to(DEVICE)

                # 开启训练
                self.net.train()
                out = self.net(image)
                # 使用自己定义的损失函数
                loss = torch.mean((tag - out) ** 2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(loss.item())
                # 收集损失
                sum_loss += loss.cpu().item()
            avg_loss = sum_loss / len(self.train_loader)

            # 验证模型
            test_sum_loss = 0
            sum_score = 0

            self.net.eval()
            for i, (image, tag) in enumerate(self.test_loader):
                image, tag = image.to(DEVICE), tag.to(DEVICE)
                test_y = self.net(image)
                test_loss = torch.mean((tag - test_y) ** 2)
                test_sum_loss += test_loss.item()

                # 模型的准确率
                idx = torch.argmax(test_y, dim=1)
                label = torch.argmax(tag, dim=1)
                sum_score += torch.sum(torch.eq(idx, label).float())

            test_avg_loss = test_sum_loss / len(self.test_loader)
            avg_score = sum_score / len(self.test_loader)

            print(epoch, avg_loss, test_avg_loss, avg_score)
            # 收集要展示的数据
            # self.sunnaryWriter.add_scalars("loss", {"train_loss": avg_loss, "test_loss": test_avg_loss}, epoch)
            # self.sunnaryWriter.add_scalar("score", avg_score, epoch)
            # 保存模型参数
            torch.save(self.net.state_dict(), f"params/{epoch}.t")


if __name__ == '__main__':
    train = Train(r"F:\2.Dataset\MNIST_DATASET")
    train()
