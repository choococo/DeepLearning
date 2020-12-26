import numpy as np

"""
工具包
"""


# box=[x1, y1, x2, y2]
def iou(box, boxes, isMin=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    inter = w * h
    if isMin:
        return inter / np.minimum(box_area, boxes_area)
    else:
        return inter / (box_area + boxes_area - inter)
