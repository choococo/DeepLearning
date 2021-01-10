# MTCNN

## 1. 数据集结构

    数据集结构：
        train:
            - 12
                -- positive
                -- part
                -- negative
                   positive.txt
                   part.txt
                   negative.txt
            - 24 
                -- positive
                -- part
                -- negative
                   positive.txt
                   part.txt
                   negative.txt
            - 48
                -- positive
                -- part
                -- negative
                   positive.txt
                   part.txt
                   negative.txt
        val:
            ...
        test:
            ...

## 2. 模型搭建
1. PNet

    
    (1) 对P网络进行改进，将MaxPooling的kerel_size=3改为kernel_size=2,这样的效果更好
    (2) 加入BatchNormal
    (3) 使用PRelu进行激活

2. RNet


    (1) 网络中加入BatchNormal
    (2) 使用PRelu激活函数进行激活

3. ONet


    (1) 网络中加入BatchNormal
    (2) 使用PRelu激活函数进行激活

## 3. 模型改进和优化
1. 模型优化
   

    （1）使用了BatchNormal
    （2）使用PRelu
    （3）卷积代替池化

2. 侦测代码优化

    
    （1）将for循环改成了矩阵运算
    （2）将r和o网络侦测代码融合，代码量更少


## 4. 训练
    
    加入过拟合自动停止判断

## 5. 模型评估


    使用R2分数对偏移量进行评估
    使用Accuracy对置信度进行评估

## 6. 实际使用

### 6.1 图片


### 6.2 摄像头


## 7. 出现的问题及优化

### 7.1 训练
1. 生成样本优化


    问题：生成效率太慢
    优化：
    （1）严格控制样本比例，正样本：部分样本：负样本=3:3:9
    （2）优化生成代码，加入多线程，加速数据生成速度


2. 训练时

    
    问题：训练一个轮次的时间太久，如果使用以往的计算一个轮次中的平均值的方式
        时间太久，训练需要很久，验证时数据收集也要很久，这样的不好直观的观察结果
    （1）调整生成样本比例，训练集：验证集：测试集=0.895:0.005:0.1
    （2）调整过拟合的判断条件，在一个轮次中分多个进行保存权重，将验证集加入到训练集中

