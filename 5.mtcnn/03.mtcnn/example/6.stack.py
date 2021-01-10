import numpy as np

a = np.array([1, 2])
b = np.array([3, 4])
c = np.array([5, 6])

ls = []
ls.append(a)
ls.append(b)
ls.append(c)

# list追加array元素后的形式
print(ls)

# 重新组合list的array元素，变成array形式,默认按照0轴组合
print(np.stack(ls, 0))
# 可以按照1轴组合
print(np.stack(ls, axis=1))
