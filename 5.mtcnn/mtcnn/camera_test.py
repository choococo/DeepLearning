# 侦测
from mtcnn.detector import Detector
import cv2
import time
from PIL import Image, ImageDraw
# from arcface.ArcFace import FaceNet, compare

if __name__ == '__main__':
    # 用摄像头框人脸
    detector = Detector()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        t1 = time.time()
        ret, frame = cap.read()
        if ret:
            start_time = time.time()
            frame = cv2.bilateralFilter(frame, 10, 30, 30)
            frames = frame[:, :, ::-1]

            image = Image.fromarray(frames, 'RGB')
            imDraw = ImageDraw.Draw(image)
            boxes = detector.detect(image)
            len = 0.2
            for box in boxes:  # 多个框，没循环一次框一个人脸
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                px1 = int(box[5])
                py1 = int(box[6])
                px2 = int(box[7])
                py2 = int(box[8])
                px3 = int(box[9])
                py3 = int(box[10])
                px4 = int(box[11])
                py4 = int(box[12])
                px5 = int(box[13])
                py5 = int(box[14])
                print(box[4])
                # imDraw.rectangle((x1, y1, x2, y2), outline='red', width=3)
                imDraw.ellipse((px1, py1, px1 + 3, py1 + 3), fill='red')
                imDraw.ellipse((px2, py2, px2 + 3, py2 + 3), fill='red')
                imDraw.ellipse((px3, py3, px3 + 3, py3 + 3), fill='red')
                imDraw.ellipse((px4, py4, px4 + 3, py4 + 3), fill='red')
                imDraw.ellipse((px5, py5, px5 + 3, py5 + 3), fill='red')

                cv2.circle(frame, center=(px1, py1), radius=2, color=(0, 0, 255), thickness=3)
                cv2.circle(frame, center=(px2, py2), radius=2, color=(0, 0, 255), thickness=3)
                cv2.circle(frame, center=(px3, py3), radius=2, color=(0, 0, 255), thickness=3)
                cv2.circle(frame, center=(px4, py4), radius=2, color=(0, 0, 255), thickness=3)
                cv2.circle(frame, center=(px5, py5), radius=2, color=(0, 0, 255), thickness=3)
                w, h = x2 - x1, y2 - y1
                # print("conf:", box[4])  # 置信度
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))

                cv2.line(frame, (x1, y1), (int(x1 + len * w), int(y1)), color=(255, 255, 255), thickness=2)
                cv2.line(frame, (x1, y1), (x1, int(y1 + len * h)), color=(255, 255, 255), thickness=2)
                cv2.line(frame, (x2, y1), (int(x2 - len * w), int(y1)), color=(255, 255, 255), thickness=2)
                cv2.line(frame, (x2, y1), (int(x2), int(y1 + len * h)), color=(255, 255, 255), thickness=2)

                cv2.line(frame, (x1, y2), (int(x1 + len * w), int(y2)), color=(255, 255, 255), thickness=2)
                cv2.line(frame, (x1, y2), (x1, int(y2 - len * h)), color=(255, 255, 255), thickness=2)
                cv2.line(frame, (x2, y2), (int(x2 - len * w), int(y2)), color=(255, 255, 255), thickness=2)
                cv2.line(frame, (x2, y2), (int(x2), int(y2 - len * h)), color=(255, 255, 255), thickness=2)
            t2 = time.time()
            fps = round(1 / (t2 - t1), 2)
            cv2.putText(frame, f"fps={str(fps)}", (50, 50),5, 3, (0, 255, 255), 1, lineType=cv2.LINE_AA)
            print("fps:", 1 / (t2 - t1))
                # 1.将框到的人脸图片裁剪出来
                # 2. 放入到人脸特征提取器中进行特征提取
                # 3. 取出数据库中的特征向量
                # 4. 进行相似度比较
            end_time = time.time()
            # print(end_time - start_time)
        cv2.imshow('MTCNN', frame)
        cv2.waitKey(10)
