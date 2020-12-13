import numpy as np

"""
LN是针对深度网络的某一层的所有神经元进行计算，计算一个批次中(在一个图片上进行计算)
LN与BN区别在与：
    (1)LN中同层神经元输入拥有相同的均值和方差，不同输入样本拥有不同的均值和方差
    (2)所以，LN不依赖于batch的大小和输入sequence的深度，因此可以用于batch size为1和
    RNN中对边长的输入sequence的normalize操作
LN 在RNN上的效果比较明显，但是在CNN上不如BN
"""


# 用在RNN上
def ln_rnn(x, batch_size, sequence):  # [N, S, V]
    eps = 1e-5

    output = (x - x.mean(1)[:, None]) / np.sqrt((x.var(1)[:, None] + eps))
    output = sequence[None, :] * output + batch_size[None, :]
    return output

# 在NCHW的中
def LayerNorm(x, gamma, beta):
    # x [N, C, H, W]
    results = 0.
    eps = 1e-5

    x_mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
    x_var = np.var(x, axis=(1, 2, 3), keepdims=True)

    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta

    return results


