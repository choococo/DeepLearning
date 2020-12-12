from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from conv_net import Conv_Net_V1
import torch
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda:0"
summaryWriter = SummaryWriter("Conv_Net_V1_logs")

train_dataset = datasets.CIFAR10(root="D:\data\CIFAR10",train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.CIFAR10(root="D:\data\CIFAR10",train=False,transform=transforms.ToTensor(),download=False)

train_loader = DataLoader(train_dataset,500,True)
test_loader = DataLoader(test_dataset,100,True)

#实例化网络
net = Conv_Net_V1().to(DEVICE)
#创建优化器
opt = torch.optim.Adam(net.parameters())
#创建损失函数(均方差损失)
loss_fun= torch.nn.MSELoss()

for epoch in range(10000):
    sum_loss = 0
    for i, (img, label) in enumerate(train_loader):
        img = img.to(DEVICE)
        # 对标签做one-hot
        label = one_hot(label, 10).to(DEVICE).float()

        out = net(img)
        loss = loss_fun(out, label)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sum_loss += loss.cpu().detach().item()

    avg_loss = sum_loss / len(train_loader)

    # 验证模型
    test_sum_loss = 0
    sum_score = 0
    for i, (img, label) in enumerate(test_loader):
        img = img.to(DEVICE)
        # 对标签做one-hot
        label = one_hot(label, 10).to(DEVICE).float()

        test_out = net(img)

        test_loss = loss_fun(test_out, label)
        test_sum_loss += test_loss

        # 计算模型的测试正确率
        predict_tags = torch.argmax(test_out, dim=1)
        label_tags = torch.argmax(label, dim=1)
        sum_score += torch.sum(torch.eq(predict_tags, label_tags).float())
    test_avg_loss = test_sum_loss / len(train_loader)
    score = sum_score / len(test_dataset)

    print(epoch, avg_loss, test_avg_loss.item(), score.item())

    # 收集要展示的数据
    summaryWriter.add_scalars("loss", {"train_loss": avg_loss, "test_loss": test_avg_loss}, epoch)
    summaryWriter.add_scalar("score", score, epoch)
    # 保存模型参数
    # torch.save(net.state_dict(), f"params/{epoch}.t")
