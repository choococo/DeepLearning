import numpy as np


def BatchNorm(x, gamma=None, beta=None, bn_param=None):
    # x [N, C, H, W]
    x_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    return x_mean


if __name__ == '__main__':
    np.random.seed(0)
    x = np.random.randn(2, 2, 2, 2)
    print(x)
    # print(np.mean(x[:, 0:1, ...]))
    #
    # print(np.mean(x[:, 1:2, ...]))
    print(BatchNorm(x))
