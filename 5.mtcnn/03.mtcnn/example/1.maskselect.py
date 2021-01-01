import torch
import numpy as np

"""
torch中使用mask_selected()
    在[N, 4]的数据找到[N, 1]的掩码, 最终得到的值是[M] 将数据扯平
torch中的大于、小于、大于等于、小于等于 等于
    gt(greater than) lt(less than) ge(greater than or equal to) le(less equal) eq(equal)
"""

seed = torch.manual_seed(0)
cls_label = torch.randint(0, 3, (10, 1))
print(cls_label.shape)
offset_label = torch.randn(10, 4)
print(offset_label.shape)

'方法一：'
mask_cls = torch.lt(cls_label, 2)  # [M, 1] 是一个二维的数据，如果用索引的方法需要降维取值
print(mask_cls.shape)  # torch.Size([10, 1])
# exit()
cls = torch.masked_select(cls_label, mask_cls)
print(cls, cls.shape)  # tensor([0, 0, 1, 0, 1, 1, 1, 0]) torch.Size([8])

mask_offset = torch.gt(cls_label, 0)
print(mask_offset.shape)  # torch.Size([10, 1])
offset = torch.masked_select(offset_label, mask_offset)
print(offset, offset.shape)
# 这里数据的顺序没有发生改变，因此可以，与下面相同
print(offset.reshape(-1, 4), offset.reshape(-1, 4).shape)

print("-----------------------------")
'方法二：使用传统方式'
mask_cls = cls_label[:, 0] < 2
cls = cls_label[mask_cls]
print(cls.shape)
mask_offset = cls_label[:, 0] > 0
offset = offset_label[mask_offset]
print(offset, offset.shape)  # torch.Size([6, 4])

from sklearn.metrics import r2_score
np.random.seed(0)
a = np.random.randn(10, 4)
b = np.random.randn(10, 4)
print(a)
a_ = a.reshape(-1)
b_ = b.reshape(-1)
print(r2_score(a, b))
print(r2_score(a_, b))
