import torch
from PIL import Image, ImageDraw
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# box = [c, x1, y1, x2, y2]
def ious(box, boxes, isMin=False):
    box_area = (box[3] - box[1]) * (box[4] - box[2])
    boxes_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])

    xx1 = torch.max(box[1], boxes[:, 1])
    yy1 = torch.max(box[2], boxes[:, 2])
    xx2 = torch.min(box[3], boxes[:, 3])
    yy2 = torch.min(box[4], boxes[:, 4])

    w = torch.clamp(xx2 - xx1, min=0)
    h = torch.clamp(yy2 - yy1, min=0)

    inter = w * h
    ovr = inter / (box_area + boxes_area - inter)
    return ovr


def nms(boxes, thresh=0.3, isMin=True):
    if boxes.shape[0] == 0:
        return torch.tensor([])

    # 1，先根据置信度进行排序
    _boxes = boxes[torch.argsort(boxes[:, 0], descending=True)]  # descending 递减的
    # 2. 然后取出第一个，进行保存
    r_boxes = []
    while _boxes.shape[0] > 1:
        a_box = _boxes[0]  # 保留第一个框
        r_boxes.append(a_box)
        b_boxes = _boxes[1:]
        iou = ious(a_box, b_boxes, isMin)
        _boxes = b_boxes[iou < thresh]
    if _boxes.shape[0] > 0:  # 如果还有剩下的
        r_boxes.append(_boxes[0])
    return torch.stack(r_boxes, dim=0)


def convert_to_416x416(image, scale_side=416, process_path=None, index=None):
    # image = Image.open(r"G:\5分类/1.jpg")
    # 获得图片原始宽高
    ow, oh = image.size
    img = image.copy()
    # 根据最大边进行缩放，图像只会缩小，不会变大
    img.thumbnail((scale_side, scale_side))  # 这个不需要返回值，直接在原图上进行了修改
    w, h = img.size
    # 缩放比例=缩放的框/原始框  (缩放比例是一个小数)
    s_w, s_h = w / ow, h / oh
    s = (s_h + s_w) / 2
    # print(s)
    bg_img = Image.new("RGB", (scale_side, scale_side), (0, 0, 0))

    bg_img.paste(img, (0, 0))
    if process_path is not None:
        bg_img.save(f"{process_path}/{index}.jpg")
    return w, h, s, bg_img


def center_to_left_and_right(image, bboxes):
    draw = ImageDraw.Draw(image)
    for box in bboxes:
        box = [float(i) for i in box]
        cls, cx, cy, w, h = box
        # cx, cy, w, h = 123, 202, 126, 118

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = x1 + w
        y2 = y1 + h
        print(x1, y1, x2, y2)

        draw.rectangle((x1, y1, x2, y2), fill=None, outline='red', width=2)
    # image.show()
    plt.imshow(image)
    plt.axis("off")
    plt.pause(1)


def show_process_img():
    label_txt = r"F:\workspace\7.YOLO\01.MyYolov3\data\lable3.txt"
    save_path = r"F:\Dataset03\VOC\process_img"
    with open(label_txt, 'r') as f:
        for i in f.readlines():
            strs = i.strip().split()
            img_name = strs[0]
            print(img_name)
            img = Image.open(os.path.join(save_path, img_name))

            _boxes = np.array(strs[1:])
            print(_boxes)
            boxes = np.stack(np.split(_boxes, len(_boxes) // 5))
            print(boxes)
            center_to_left_and_right(img, boxes)
            # exit()


if __name__ == '__main__':
    # 1.
    # bs = torch.tensor([[1, 1, 10, 10, 40, 8], [1, 1, 9, 9, 10, 9], [9, 8, 13, 20, 15, 3], [6, 11, 18, 17, 13, 2]])
    # # print(bs[:,3].argsort())
    # print(nms(bs))

    # 2.
    # img_path = r"G:\5class_2_group\img"
    # save_path = r"G:\5class_2_group\process_img"
    # #
    # img_total = os.listdir(img_path)
    # length = len(img_total)
    # for i, img_name in enumerate(tqdm(img_total)):
    #     # print(i, img_name)
    #     img = Image.open(os.path.join(img_path, img_name))
    #     # convert_to_416x416(img, 416, save_path, index=i)
    #     convert_to_416x416(img, 416, save_path)
    #     # exit()
    # # exit()

    # 3.
    show_process_img()
