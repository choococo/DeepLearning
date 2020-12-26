import numpy as np
import utils.tools as tools

"""
mask的研究的最新的使用方法：
    在写MTCNN样本生成的时，用于筛选满足置信度的条件的bbox
"""
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

part_box = crop_box[(0.3 < iou_p) & (iou_p < 0.67)]
