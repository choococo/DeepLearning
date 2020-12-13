import numpy as np

"""
主要针对BN对小batch size效果差，GN将channel方向分组，然后每个group内做归一化，这样与
batch size无关，不受其约束
"""


def GroupNorm(x, gamma=1.0, beta=0., G=16):
    # x [N, C, H, W]
    results = 0.
    eps = 1e-5

    N, C, H, W = x.shape
    x = x.reshape(N, G, C // G, H, W)
    x_mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    x_var = np.var(x, axis=(2, 3, 4), keepdims=True)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    # x_normalized = x_normalized.reshape(N, C, H, W)
    return gamma * x_normalized + beta


x = np.random.randn(2, 32, 4, 4)
print(GroupNorm(x).shape)
