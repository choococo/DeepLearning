import numpy as np

"""
batch  normalization 计算过程
    (1)沿着通道计算每个batch的均值
    (2)沿着通道计算每个batch的方差
    (3)对x做归一化，x'=(x-miu)/sqrt(var)
    (4)加入缩放和平移变量gamma和beta，归一化后的值y=gamma*x‘+beta
    
batch Normalization的缺点：
    (1)对batch_size的大小比较敏感，由于每次计算均值和方差都是在一个batch上，所以如果
        batch_size太小则计算的方差和均值不足以代表整个数据分布
    (2)BN 实际使用时需要计算并且保存某一个神经网络batch的均值和方差等统计信息，对于一个
    固定的深度前向神经网络(DNN,CNN)使用BN，很方便；但是对于RNN来说，由于sequence的长度
    是不一致的，换句话说RNN的深度是不固定的，不同的time-step需要保存不同的statics特征，
    可能存在一个特殊sequence比其他sequence长很多，这样在训练时，计算很麻烦
    
"""


def BatchNorm(x, momentum=None, gamma=None, beta=None, bn_param=None):
    # x [N, C, H, W]
    running_mean = bn_param['running_mean']
    running_var = bn_param['running_var']
    results = 0.
    eps = 1e-5

    x_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    x_var = np.var(x, axis=(0, 2, 3), keepdims=True)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)

    results = gamma * x_normalized + beta

    # 因为在测试的时是当个图片测试，这里保留训练时的均值和方差，用在后面测试时用
    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return results, bn_param


if __name__ == '__main__':
    np.random.seed(0)
    x = np.random.randn(2, 2, 2, 2)
    print(x)
    # print(np.mean(x[:, 0:1, ...]))
    #
    # print(np.mean(x[:, 1:2, ...]))
    print(BatchNorm(x))
