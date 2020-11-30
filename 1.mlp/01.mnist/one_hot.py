import numpy as np
import torch

'one-hot使用方法：最简单'
num_classes = 10
arr = [1, 3, 4, 5]
one_hots = np.eye(num_classes)[arr]
print(one_hots)

arr1 = torch.LongTensor(arr)
print(arr1)
one_hots = torch.eye(num_classes)[arr]
print(one_hots)

arr2 = np.array(arr)
print(arr2)
one_hots = np.eye(num_classes)[arr]
print(one_hots)

'one-hot使用方法二：'
one_hot = np.zeros([len(arr2), num_classes])
for i, j in enumerate(arr):
    one_hot[i][int(j)] = 1
print(one_hot)

'one-hot使用方式三：'
one_hot = torch.zeros((len(arr1), num_classes)).scatter_(dim=1, index=arr1[:, None], value=1)
print(one_hot)