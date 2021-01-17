import torch
import torch.nn as nn


def center_loss(feature, label, lambdas):
    center = nn.Parameter(torch.randn(int(max(label).item() + 1), feature.shape[1]), requires_grad=True).cuda()
    # print(center)
    center_exp = center.index_select(dim=0, index=label.long())
    # print(center_exp)

    count = torch.histc(label, bins=int(max(label).item() + 1), min=0, max=int(max(label).item()))
    # print(count)
    count_exp = count.index_select(dim=0, index=label.long())
    # print(count_exp)

    # center loss
    loss = lambdas / 2 * torch.mean(torch.div(torch.sum(torch.pow(feature - center_exp, 2), dim=1), count_exp))
    return loss


if __name__ == '__main__':
    data = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 8], [6, 5]], dtype=torch.float32).cuda()
    label = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32).cuda()
    loss = center_loss(data, label, 2)
    print(loss)
