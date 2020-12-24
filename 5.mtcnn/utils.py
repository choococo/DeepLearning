import numpy as np
import matplotlib.pyplot as plt


#       0   1   2   3
# box [x1, y1, x2, y2]
def iou(box1, box2):
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    x1 = np.maximum(box1[0], box2[0])
    y1 = np.maximum(box1[1], box2[1])
    x2 = np.minimum(box1[2], box2[2])
    y2 = np.minimum(box1[3], box2[3])

    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)

    inter_area = w * h

    return inter_area / (box1_area + box2_area - inter_area)


# def draw_rect(box):
#     fig, ax = plt.subplots()
#     # box = [int(i) for i in box]
#     for i in box:
#         rect = plt.Rectangle((i[0], i[1]), i[2] - i[0], i[3] - i[1], fill=False, color='red', linewidth=2)
#         ax.add_patch(rect)
#     plt.axis("off")
#     plt.show()
def draw_rect(box1, box2):
    fig, ax = plt.subplots()
        # 长方形
    rect1 = plt.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1], fill=False, color='red', linewidth=2)
    ax.add_patch(rect1)
    rect2 = plt.Rectangle((box2[0], box2[1]), box2[2] - box2[0], box2[3] - box2[1], fill=False, color='red', linewidth=2)
    ax.add_patch(rect2)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    box1 = np.array([10, 10, 20, 20])
    box2 = np.array([0, 0, 5, 5])
    print(iou(box1, box2))
    draw_rect(box1, box2)
