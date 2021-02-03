import torch
from utils import tools
from detector import Detector, img_transform
import os
from PIL import Image, ImageDraw
import cfg
import cv2
import numpy as np

if __name__ == '__main__':
    # checkpoints = r"F:\workspace\7.YOLO\01.MyYolov3\checkpoints\net_yolo_72.pt"
    checkpoints = r"checkpoints\net_yolo_500.pt"
    img_path = r"G:\3class_2_group\img"
    detector = Detector(save_path=checkpoints)
    cap = cv2.VideoCapture(r"F:\workspace\7.YOLO\01.MyYolov3\data\images\3.jpg")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            _, _, s, pro_image = tools.convert_to_416x416(image, 416)
            img_tensor = img_transform(pro_image)[None, ...].cuda()
            out_value = detector(img_tensor, 0.6, cfg.ANCHORS_GROUP).cpu().detach()
            boxes = []

            for j in range(cfg.CLASS_NUM):
                classify_mask = (out_value[..., -1] == j)
                _boxes = out_value[classify_mask]
                if _boxes.shape[0] != 0:
                    boxes.append(tools.nms(_boxes))
            for box in boxes:
                try:
                    for i in box:
                        c, xx1, yy1, xx2, yy2 = i[0:5]
                        print(c, xx1, yy1, xx2, yy2)
                        draw.rectangle((xx1 / s, yy1 / s, xx2 / s, yy2 / s))
                except:
                    continue
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow("", frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
