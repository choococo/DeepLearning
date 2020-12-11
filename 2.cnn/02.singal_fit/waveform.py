import numpy as np
import matplotlib.pyplot as plt
import torch

"""
制造数据，50-100中的数据，随机生成50个数据作为一个周期的波形，然后总共6个周期
"""


# def train_data():
#     np.random.seed(0)
#     x = np.random.uniform(50, 100, (50,))
#
#     y_list = []
#     for j in range(30):
#         for i in x:
#             y = i + np.random.uniform(-2, 2)
#             y_list.append(y)
#
#     with open("waveform_test.data", "w") as f:
#         for i in range(len(y_list)):
#             print(i)
#             f.write(str(y_list[i]))
#             f.write("\n")
#             f.flush()
#     # plt.plot([i for i in range(len(y_list))], y_list)
#     # plt.show()

def gen_test(root):
    np.random.seed(0)
    x = np.random.uniform(0, 10, (10,))
    y_list = []
    for j in range(30):
        for i in x:
            y = i + np.random.uniform(-0.5, 0.5)
            y_list.append(y)

    with open(root, "w") as f:
        for i in range(len(y_list)):
            print(i)
            f.write(str(y_list[i]))
            f.write("\n")
            f.flush()


path1 = r"waveform.data"
path2 = r"waveform_test.data"
# gen_test(path1)
train_data = np.loadtxt(path1)

with open(path2, "w") as f:
    test_list = []
    for i in train_data:
        i = i + np.random.uniform(-0.5, 0.5)
        f.write(str(i))
        f.write("\n")
        f.flush()
