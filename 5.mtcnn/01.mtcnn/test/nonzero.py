import torch
import numpy as np

a = torch.tensor([[1, 2], [3, 4], [5, 6]])

print(a > 3)
print(a[a > 3])

print(torch.nonzero(a > 3))
