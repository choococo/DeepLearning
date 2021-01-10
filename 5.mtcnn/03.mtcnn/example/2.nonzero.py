import torch
import numpy as np

"""
torch中的nonzero() 返回的一个组合好的行和列
    tensor([[0, 4],
            [1, 0],
            [2, 3],
            [5, 0],
            [5, 4]])
numpy中的where() 返回的一个行的索引 和列的索引
    [0 3 4] [0 0 0]
"""

# cls[H,W] offset[4,H,W]
seed = torch.manual_seed(0)
cls = torch.randn(6, 5)
offset = torch.randn(4, 6, 5)
print(cls)
print(offset)

idxs = torch.nonzero(cls > 0.6)
print(idxs)
# 用API会比较好，上面的会警告
idxs = torch.nonzero(torch.gt(cls, 0.6), as_tuple=True)  # 这个为True是为了符合np.where中的形式，np中是分开的
# (tensor([0, 1, 2, 5, 5]), tensor([4, 0, 3, 0, 4]))
print(idxs)
idxs = torch.nonzero(torch.gt(cls, 0.6))
print(idxs)
"""
tensor([[0, 4],
        [1, 0],
        [2, 3],
        [5, 0],
        [5, 4]])
tensor([[0, 4],
        [1, 0],
        [2, 3],
        [5, 0],
        [5, 4]])
"""

cls_label = np.random.randn(6, 1)
offset_label = np.random.randn(6, 4)
print(cls_label)

idx, idy = np.where(cls_label > 0.3)
print(idx, idy)
idx, idy = np.nonzero(cls_label > 0.3)
print(idx, idy)
cls_label = torch.randn(6, 1)
offset_label = torch.randn(6, 4)
idx = torch.nonzero(cls > 0.3)
print(idx)

print("-------------------")
idxs = np.array([0, 1, 3, 4, 5])
boxes = np.random.randn(8, 5)
cls = np.random.randn(8, 1)
x1 = boxes[idxs][:, 0]
y1 = boxes[idxs][:, 1]
x2 = boxes[idxs][:, 2]
y2 = boxes[idxs][:, 3]
cls = cls[idxs][:, 0]
# print(x1)
# print(y1)
# print(x2)
# print(y2)
# print(cls.reshape(-1))
# print(cls[:, 0])  # 和上面reshape的用法相同
# print(boxes[idxs])
a = np.stack([x1, y1, x2, y2, cls], axis=1)
print(a)
