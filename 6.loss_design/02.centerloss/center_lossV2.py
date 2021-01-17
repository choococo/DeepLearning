import torch
import torch.nn as nn


class CenterLoss(nn.Module):

    def __init__(self, lambdas, feature_num=2, class_num=10):
        super(CenterLoss, self).__init__()
        self.lambdas = lambdas
        self.center = nn.Parameter(torch.randn(class_num, feature_num), requires_grad=True)
        # self.center = torch.randn(class_num, feature_num, requires_grad=True)

    def forward(self, feature, label):
        center_exp = self.center.index_select(dim=0, index=label.long())
        count = torch.histc(label, bins=int(max(label).item() + 1), min=0, max=int(max(label).item()))
        count_exp = count.index_select(dim=0, index=label.long())
        loss = self.lambdas / 2 * torch.mean(torch.div(torch.sum(torch.pow(feature - center_exp, 2), dim=1), count_exp))
        return loss


if __name__ == '__main__':
    data = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 8], [6, 5]], dtype=torch.float32)
    label = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32)
    center_loss = CenterLoss(2, 2, 5)
    print(center_loss(data, label))
    print(list(center_loss.parameters()))
    opt = torch.optim.SGD(center_loss.parameters(), 0.1)
    opt.zero_grad()
    opt.step()
