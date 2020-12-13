import numpy as np

"""
Switchable Normalization论文作者认为：
    第一：归一化虽然提高模型的泛华能力，然而归一化层的操作是人工设计的。在实际的应用中，解决不同的问题的
    原则上需要设计不同的归一化操作，并且没有一个通用化方法能够解决所有应用的问题
    第二: 一个神经网络往往包含几十个归一化层，通常这些归一化层都使用同样的归一化操作，因为手工为每一个归
    一化层设计操作需要大量的试验
    因此作者提出自适配归一化方法SN来解决上述问题。与强化学习不同，SN使用可微分学习，为一个神经网络的中每一个
    归一化层确定合适的归一化操作
"""


def SwitchableNorm(x, gamma, beta, w_mean, w_var):
    # x [N, C, H, W]
    results = 0.
    eps = 1e-5

    mean_in = np.mean(x, axis=(2, 3), keepdims=True)
    var_in = np.var(x, axis=(2, 3), keepdims=True)

    mean_ln = np.mean(x, axis=(1, 2, 3), keepdims=True)
    var_ln = np.var(x, axis=(1, 2, 3), keepdims=True)

    mean_bn = np.mean(x, axis=(0, 2, 3), keepdims=True)
    var_bn = np.var(x, axis=(0, 2, 3), keepdims=True)

    mean = w_mean[0] * mean_in + w_mean[1] * mean_ln + w_mean[2] * mean_bn
    var = w_var[0] * var_in + w_var[1] * var_ln + w_var[2] * var_bn

    x_normalized = (x - mean) / np.sqrt(var + eps)
    results = gamma * x_normalized + beta

    return results
