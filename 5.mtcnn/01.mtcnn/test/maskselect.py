import torch

a = torch.tensor([1, 2, 3, 4, 5])

print(a < 4)
print(torch.lt(a, 4))  # lt gt eq le ge

print(a[a < 4])
print(torch.masked_select(a, a < 4))

print(torch.nonzero(a < 4))
