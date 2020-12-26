import numpy as np

"""
cat操作是一个很好地操作，可以很好的对高维的数据进行拼接等操作，numpy和torch包中都有API
应用地方：
    （1）曾经在复现MTCNN代码的时候用到：主要用到的地方，1）在制作数据样本时，使用
"""
seed = np.random.seed(0)
a = np.random.randint(0, 10, (10, 1))
b = np.random.randint(0, 10, (10, 1))
print(a)
print(b)
c = np.concatenate([a, b], axis=0)  # 这里在0轴的时候，相当于列表的追加
print(c)
d = np.concatenate([a, b], axis=1)  # 对a,b进行一对一的组合
print(d)

"""
[[5 7]
 [0 6]
 [3 8]
 [3 8]
 [7 1]
 [9 6]
 [3 7]
 [5 7]
 [2 8]
 [4 1]]
"""