import numpy as np

a = np.array([[5, 4], [3, 2], [1, 8], [7, 9]])

# 按照第1轴的第0个元素从大到小排序
# a.sort(axis=0)
# print(a)

# sort没有返回值，会直接改变当前值
# index1 = (a[:,0]).sort()
# print(a)
# print(a[index1])

# argsort的返回值为索引，不会改变当前值
# index1 = (-a[:,0]).argsort()
# print(a)
# print(index1)
# print(a[index1])

boxes = np.array([[12, 48, 26, 15, 0.6], [39, 25, 9, 32, 0.8], [54, 21, 59, 10, 0.5], [65, 28, 94, 14, 0.9]])
_boxes = boxes[(-boxes[:, 4]).argsort()]
print(_boxes)
