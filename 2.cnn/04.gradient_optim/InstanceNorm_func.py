import numpy as np

"""
BN注重对每个batch进行归一化，保证数据分布一致，因为判别模型中结果取决与数据整体的分布
但是图像风格迁移中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合推向风格化中
因而对HW做归一化，可以加速模型收敛，并且保持图象实例之间的独立
Instance Normalization：对HW进行归一化
"""

def InstaceNorm(x, gamma=1, beta=0):
    # x [N, C, H, W]
    results = 0.
    eps = 1e-5

    x_mean = np.mean(x, axis=(2, 3), keepdims=True)
    x_var = np.mean(x, axis=(2, 3), keepdims=True)

    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta

    return results


x = np.random.randn(2, 3, 4, 4)
print(InstaceNorm(x).shape)