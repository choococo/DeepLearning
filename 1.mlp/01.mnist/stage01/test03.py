import torch

a = torch.randn(4,5)
print(a)
print(torch.max(a,1)[0].shape)