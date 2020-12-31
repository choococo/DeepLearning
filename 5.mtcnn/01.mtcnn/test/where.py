import numpy as np

a = np.array([8, 2, 7, 5, 1, 4])

# bool值
print(a < 5)
# 值
print(a[a < 5])

# 索引
print(np.where(a < 5))
# 值
print(a[np.where(a < 5)])
