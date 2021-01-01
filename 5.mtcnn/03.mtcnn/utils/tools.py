import numpy as np
import matplotlib.pyplot as plt

"""
工具包
"""


#   0   1   2   3   4
# [x1, y1, x2, y2, conf]
def iou(box, boxes, isMin=False):  # 计算置信度
    """
    计算iou
    :param box: 一个框
    :param boxes: 多个框
    :param isMin: 是否是交小IOU
    :return: iou矩阵
    """
    box_area = (box[2] - box[0]) * (box[3] - box[1])  # 计算框的面积
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # 计算多个框的面积

    x1 = np.maximum(box[0], boxes[:, 0])  # 两个框中左上角的最大值
    y1 = np.maximum(box[1], boxes[:, 1])  # 两个框中左上角的最大值
    x2 = np.minimum(box[2], boxes[:, 2])  # 两个框中右下角的最小值
    y2 = np.minimum(box[3], boxes[:, 3])  # 两个框中右下角的最小值

    w = np.maximum(0, x2 - x1)  # 保证交集的宽最小为0,不想交
    h = np.maximum(0, y2 - y1)  # 保证交集的宽最小为0,不想交

    inter = w * h  # 计算交集的面积

    if isMin:  # 计算交小IOU,这个通常使用在ONet中,用于去除大框套小框的操作
        return inter / np.minimum(box_area, boxes_area)
    else:  # 普通的IOU
        return inter / (box_area + boxes_area - inter)


#   0   1   2   3   4
# [x1, y1, x2, y2, conf]
def nms(boxes, thresh=0.3, isMin=False):
    """
    计算NMS
    :param boxes: 多个框
    :param thresh: 阈值
    :param isMin: 是否为交小IOU
    :return: 返回根据阈值筛选后的框
    """
    if boxes.shape[0] == 0:  # 常规判空操作,防止框中没有值
        return np.array([])  # 直接染回空numpy 数组

    keep_boxes = []  # 最终需要保留的框
    # 根据置信度进行排序 argsort:从小到大排序
    # boxes_sort = boxes[boxes[:, 4].argsort()[::-1]]  # 方法1
    boxes_sort = boxes[(-boxes[:, 4]).argsort()]  # 方法2
    while len(boxes_sort) > 1:  # 如果排序后的框有两个值
        box_ = boxes_sort[0]  # 每次保留置信度最大的第一个框
        keep_boxes.append(box_)  # 添加到列表中

        remainder_box = boxes_sort[1:]  # 剩下的框
        # 计算IOU
        iou_ = iou(box_, remainder_box, isMin)  # 计算IOU
        # print(iou_ < thresh)  # [False  True  True]
        # print(np.where(iou_ < thresh))  # (array([1, 2], dtype=int64),) 满足条件的索引

        boxes_sort = remainder_box[iou_ < thresh]  # 重新对数据进行操作
    if boxes_sort.shape[0] > 0:  # 如果是最后一个也需要添加
        keep_boxes.append(boxes_sort[0])
    return np.stack(keep_boxes, axis=0)  # 使用stack对[array(...), array(...),...]这种的进行组合


#   0   1   2   3   4
# [x1, y1, x2, y2, conf]
def convert_to_square(bbox):
    """
    在输入下一个网络之前，将P/R/O网络输出的转成正方形
    :param bbox: 多个框
    :return: 转成正方形的框
    """
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    max_side = np.maximum(w, h)
    square_bbox[:, 0] = bbox[:, 0] + w / 2 - max_side / 2  # 先计算中心点-最长边的一半
    square_bbox[:, 1] = bbox[:, 1] + h / 2 - max_side / 2
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side

    return square_bbox


def show_rect(boxes):
    fig, ax = plt.subplots()
    for i in boxes:
        x1, y1, x2, y2 = i[:4]
        rect = plt.Rectangle((x1, y1), width=(x2 - x1), height=(y2 - y1), fill=False, color="red", linewidth=2)
        ax.add_patch(rect)
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    # bs = np.array([[1, 1, 10, 10, 0.98], [1, 1, 9, 9, 0.8], [9, 8, 13, 20, 0.7], [6, 11, 18, 17, 0.85]])
    bs = np.array([[1, 1, 10, 10, 40], [1, 1, 9, 9, 10], [9, 8, 13, 20, 15], [6, 11, 18, 17, 13]])
    print(nms(bs))
    show_rect(bs)
    show_rect(nms(bs, 0.25))
