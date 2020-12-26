import numpy as np
import torch

"""
种子的使用：
    （1）使用种子随机数seed可以保持当前的随机保持不变
    （2）是一个数保持不变，其他的数保持变化
"""


def seed_func1():
    seed = np.random.seed(0)
    a = np.random.randint(10)
    print(a)  # 种子函数可保持许多的内容


def seed_func2():
    np.random.seed(0)
    a = np.random.randint(0, 10, (10,))
    print(a)
    for i in range(2):

        b = np.random.randint(0, 10, (10,))
        print(b)


if __name__ == '__main__':
    # seed_func1()
    seed_func2()
