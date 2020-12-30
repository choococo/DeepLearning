import numpy as np
import utils.tools as tools
import torch

"""
mask的研究的最新的使用方法：
    使用一：在写MTCNN样本生成的时，用于筛选满足置信度的条件的bbox
    使用二：在计算有物体的格子的坐标
"""


def mask_use1():
    box_label = np.array([95, 71, 321, 384])

    x1_ = np.random.randint(100, 200, (100, 1))
    y1_ = np.random.randint(100, 200, (100, 1))
    w_ = np.random.randint(0, 50)
    h_ = np.random.randint(0, 50)
    x2_ = x1_ + w_
    y2_ = y1_ + h_
    print(x2_.shape)  # (100, 1)

    # crop_boxes = np.concatenate([x1_, y1_, x2_, y2_], axis=1)  # 把一轴进行拼接称为2
    # print(crop_boxes.shape)  # (100, 4)
    crop_box = np.array([[75.5, 64., 354.5, 343.],
                         [66.5, 82., 345.5, 361.],
                         [62.5, 84., 341.5, 363.],
                         [48.5, 90., 327.5, 369.],
                         [58.5, 69., 337.5, 348.],
                         [75.5, 95., 354.5, 374.],
                         [93.5, 106., 372.5, 385.],
                         [59.5, 82., 338.5, 361.],
                         [56.5, 104., 335.5, 383.],
                         [58.5, 77., 337.5, 356.]])
    iou = tools.iou(box_label, crop_box)
    print(iou)
    iou_p = tools.iou(box_label, crop_box)
    # 正样本
    positive_box = crop_box[iou_p > 0.7]
    print(positive_box)

    'NB的使用1: (0.3 < iou_p) & (iou_p < 0.67)'
    part_box = crop_box[(0.3 < iou_p) & (iou_p < 0.67)]
    print(part_box)


'NB的使用2:np.nonzero()与np.where()的用法差不多'


def mask_use2():
    np.random.seed(0)
    conf = np.random.randn(1, 1, 6, 6)
    offset = np.random.randn(1, 4, 6, 6)

    # 使用一张图片
    conf = conf[0, 0]
    print(conf.shape)
    offset = offset[0]
    print(offset.shape)
    # 拿到有物体格子的索引
    # idx = np.nonzero(conf > 0.7)  # np.nonzero() 返回的一个元组，行和列
    idx = np.where(conf > 0.7)  # np.nonzero() 返回的一个元组，行和列
    print(idx)
    """
    (   array([0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4], dtype=int64), # 这个代表行索引
        array([0, 2, 3, 4, 0, 5, 0, 4, 4, 0, 4, 5], dtype=int64)  # 这个代表列索引  )
        
    (   array([0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4], dtype=int64), 
        array([0, 2, 3, 4, 0, 5, 0, 4, 4, 0, 4, 5], dtype=int64))
    """
    # 将行索引和列索引组合在一起：（1）先把最外边的元组拆开 （2）再使用zip进行组合
    idx = np.array([[idx, idy] for idx, idy in zip(*idx)])
    print(idx)

    # 拿到有物体格子的坐标的偏移量
    offset_box = offset[:, idx[:, 0], idx[:, 1]]
    print(offset_box.shape)


'NB的使用：torch.nonzero():将行和列索引组装在一起'


def mask_user3():
    seed = torch.random.manual_seed(0)
    output = torch.randn(2, 5, 6, 6)
    conf, offset = output[:, 0], output[:, 1:]
    print(conf.shape, offset.shape)  # torch.Size([2, 6, 6]) torch.Size([2, 4, 6, 6])
    idx = torch.nonzero(conf > 0.7)
    print(idx.shape)  # torch中是组合好了的，不需要自己组合
    offset_box = offset[:, idx[:, 0], idx[:, 1], idx[:, 2]]
    print(offset_box)


if __name__ == '__main__':
    # mask_use1()
    mask_use2()
    # mask_user3()
