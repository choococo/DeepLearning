# 侦测
import torch
from detector_lzy import Detector
import cv2
import time
from PIL import Image


if __name__ == '__main__':
    # 用摄像头框人脸
    detector = Detector()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            start_time = time.time()
            frames = frame[:, :, ::-1]

            image = Image.fromarray(frames, 'RGB')

            boxes = detector.detect(image)
            for box in boxes:  # 多个框，没循环一次框一个人脸
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                print("conf:", box[4])  # 置信度
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))

            end_time = time.time()
            # print(end_time - start_time)
        cv2.imshow('MTCNN', frame)
        cv2.waitKey(10)


