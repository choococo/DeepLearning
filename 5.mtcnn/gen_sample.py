import os
import traceback
import numpy as np
from PIL import Image, ImageDraw
import utils
import matplotlib.pyplot as plt
import time
import sys

# 标签路径
# choices = {0: "train", 1: "val", 2: "test"}
# label_position_path = fr"new_label/{choices[0]}_position.txt"
# label_landmark_path = fr"new_label/{choices[0]}_landmark.txt"
# img_path = r"image"
#
# save_path = fr"F:\2.Dataset\mtcnn_dataset\testing\saving\{choices[2]}"


def gen_data(img_path, label_position_path, label_landmark_path, save_path):
    t1 = time.time()
    for face_size in [12, 24, 48]:

        # print("gen %i image" % face_size)  # %i: 十进制数占位符

        # 创建样本目录
        positive_image_dir = os.path.join(save_path, str(face_size), "positive")
        negative_image_dir = os.path.join(save_path, str(face_size), "negative")
        part_image_dir = os.path.join(save_path, str(face_size), "part")

        for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # 样本标签text存储路径
        positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
        negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
        part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

        # 计数器
        positive_count = 0
        negative_count = 0
        part_count = 0

        try:
            positive_anno_file = open(positive_anno_filename, "w")  # 以写入的形式打开txt文档
            negative_anno_file = open(negative_anno_filename, "w")
            part_anno_file = open(part_anno_filename, "w")

            f_position = open(label_position_path).readlines()  # 读入人脸位置标签
            f_landmark = open(label_landmark_path).readlines()  # 读入人脸关键点坐标标签

            for i in range(len(f_position)):
                if i < 2:
                    continue
                strs_position = f_position[i].strip().split()
                strs_landmark = f_landmark[i].strip().split()

                strs_position[1:] = [int(x) for x in strs_position[1:]]
                strs_landmark[1:] = [int(x) for x in strs_landmark[1:]]

                image_filename = strs_position[0].strip()  # 图片名称
                # print(image_filename)

                x1 = float(strs_position[1])
                y1 = float(strs_position[2])
                w = float(strs_position[3])
                h = float(strs_position[4])

                x2 = float(x1 + w)
                y2 = float(y1 + h)

                fx1 = float(strs_landmark[1])
                fy1 = float(strs_landmark[2])
                fx2 = float(strs_landmark[3])
                fy2 = float(strs_landmark[4])
                fx3 = float(strs_landmark[5])
                fy3 = float(strs_landmark[6])
                fx4 = float(strs_landmark[7])
                fy4 = float(strs_landmark[8])
                fx5 = float(strs_landmark[9])
                fy5 = float(strs_landmark[10])

                if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                    continue

                cx = x1 + w * 0.5
                cy = y1 + h * 0.5

                side = np.maximum(w, h)

                img = Image.open(os.path.join(img_path, image_filename))

                img_w, img_h = img.size

                boxes = np.array([x1, y1, x2, y2])
                label_scale_num = {"positive": 3, "part": 3, "negative": 9}
                while True:
                    w_ = np.random.randint(-(img_w - w) * 0.2, (img_w - w) * 0.2)
                    h_ = np.random.randint(-h * 0.2, h * 0.2)
                    cx_ = cx + w_
                    cy_ = cy + h_
                    side_ = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.1 * min(w, h)))
                    x1_ = np.maximum(cx_ - side_ / 2, 0)
                    y1_ = np.maximum(cy_ - side_ / 2, 0)
                    x2_ = x1_ + side_
                    y2_ = y1_ + side_
                    crop_box = np.array([x1_, y1_, x2_, y2_])

                    nx1_ = np.random.randint(0, int(img_w - min(img_w, img_h) * 0.4))
                    ny1_ = np.random.randint(0, int(img_h - min(img_w, img_h) * 0.4))
                    nx2_ = nx1_ + int(min(img_w, img_h) * 0.4)
                    ny2_ = ny1_ + int(min(img_w, img_h) * 0.4)
                    crop_box_n = np.array([nx1_, ny1_, nx2_, ny2_])

                    offset_x1 = (x1 - x1_) / side_
                    offset_y1 = (y1 - y1_) / side_
                    offset_x2 = (x2 - x2_) / side_
                    offset_y2 = (y2 - y2_) / side_

                    offset_px1 = (fx1 - x1_) / side_
                    offset_px2 = (fx2 - x1_) / side_
                    offset_px3 = (fx3 - x1_) / side_
                    offset_px4 = (fx4 - x1_) / side_
                    offset_px5 = (fx5 - x1_) / side_

                    offset_py1 = (fy1 - y1_) / side_
                    offset_py2 = (fy2 - y1_) / side_
                    offset_py3 = (fy3 - y1_) / side_
                    offset_py4 = (fy4 - y1_) / side_
                    offset_py5 = (fy5 - y1_) / side_

                    # img = Image.open(os.path.join(img_path, image_filename))
                    # draw = ImageDraw.Draw(img)
                    # draw.rectangle((x1, y1, x2, y2), outline="red", width=3)

                    iou = utils.iou(crop_box, boxes)
                    iou_n = utils.iou(crop_box_n, boxes)
                    # print("==>", iou_n)
                    if iou > 0.57 and label_scale_num['positive'] > 0:
                        label_scale_num['positive'] -= 1
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                        positive_anno_file.write(
                            "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                positive_count, 1, offset_x1, offset_y1,
                                offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                        positive_anno_file.flush()
                        face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                        positive_count += 1
                    elif 0.3 < iou < 0.57 and label_scale_num['part'] > 0:
                        label_scale_num['part'] -= 1
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                        part_anno_file.write(
                            "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                part_count, 2, offset_x1, offset_y1, offset_x2,
                                offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                        part_anno_file.flush()
                        face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                        part_count += 1
                    elif iou_n < 0.1 and label_scale_num['negative'] > 0:
                        label_scale_num['negative'] -= 1
                        face_crop = img.crop(crop_box_n)
                        face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                        negative_anno_file.write(
                            "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                        negative_anno_file.flush()
                        face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                        negative_count += 1
                        # draw.rectangle((nx1_, ny1_, nx2_, ny2_), outline="blue", width=3)
                    # print(label_scale_num['positive'], label_scale_num['part'],label_scale_num['negative'])
                    if label_scale_num['positive'] == 0 and label_scale_num['part'] == 0 and label_scale_num[
                        'negative'] == 0:
                        break
                    # plt.imshow(img)
                    # plt.axis("off")
                    # plt.pause(0.3)
                    # plt.clf()


        except Exception:
            traceback.print_exc()  # 如果出现异常，就把异常打印出来

        finally:
            positive_anno_file.close()
            negative_anno_file.close()
            part_anno_file.close()
    print(time.time() - t1)
