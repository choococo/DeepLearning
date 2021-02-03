import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import time
import data_enhance as enhance
import multiprocessing

"""
数据生成
"""


def Iou(box1, box2):
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


def get_picture():
    pic_info = []
    with open("label/list_bbox_celeba.txt") as f:
        info = f.readlines()[2:]
        for data in info:
            filename = data.strip().split()[0]
            x1 = int(data.strip().split()[1])
            y1 = int(data.strip().split()[2])
            w = int(data.strip().split()[3])
            h = int(data.strip().split()[4])
            pic_info.append([filename, x1, y1, w, h])
    return pic_info


def get_sample(label_path):
    return pd.read_table(label_path, header=1, delim_whitespace=True)


def get_landmark(landmark_path):
    return pd.read_table(landmark_path, header=1, delim_whitespace=True)


def generator_sample(split_data, landmark_data, image_path, saving_path, face_size):
    # for face_size in [12, 24, 48]:
    # 创建样本目录
    # print(face_size)
    positive_image_dir = f"{saving_path}/{str(face_size)}/{'positive'}"
    negative_image_dir = f"{saving_path}/{str(face_size)}/{'negative'}"
    part_image_dir = f"{saving_path}/{str(face_size)}/{'part'}"

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 样本标签text存储路径
    positive_anno_filename = f"{saving_path}/{str(face_size)}/{'positive.txt'}"
    negative_anno_filename = f"{saving_path}/{str(face_size)}/{'negative.txt'}"
    part_anno_filename = f"{saving_path}/{str(face_size)}/{'part.txt'}"
    # part_anno_filename = f"{saving_path}/{'part.txt'}"
    # positive_anno_filename = f"{saving_path}/{'positive.txt'}"
    # negative_anno_filename = f"{saving_path}/{'negative.txt'}"

    positive_count = 0
    negative_count = 0
    part_count = 0

    positive_anno_file = open(positive_anno_filename, "w")  # 以写入的形式打开txt文档
    negative_anno_file = open(negative_anno_filename, "w")
    part_anno_file = open(part_anno_filename, "w")

    # for i in split_data:
    for i in range(len(split_data)):
        image_filename, x1, y1, w, h = split_data.iloc[i]
        # fx1, fy1, fx2, fy2, fx3, fy3, fx4, fy4, fx5, fy5 = landmark_data.iloc[i]
        # print(image_filename)
        x1 = float(x1)
        y1 = float(y1)
        w = float(w)
        h = float(h)
        x2 = int(x1 + w)
        y2 = float(y1 + h)
        # image_filename, x1, y1, w, h = i
        # # print(image_filename)
        # x1 = float(x1)
        # y1 = float(y1)
        # w = float(w)
        # h = float(h)
        # x2 = int(x1 + w)
        # y2 = int(y1 + h)

        if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0 or min(w, h) * min(w, h) / (w * h) < 0.7:
            print(image_filename)
            continue

        cx = x1 + w * 0.5
        cy = y1 + h * 0.5

        img = Image.open(f"{image_path}/{image_filename}")
        t1 = time.time()
        img_w, img_h = img.size
        side_ = np.random.choice([w, h])
        box_label = np.array([x1, y1, x2, y2])
        label_scale_num = {"positive": 3, "part": 3, "negative": 9}
        while True:
            # img = Image.open(os.path.join(image_path, image_filename))
            # draw = ImageDraw.Draw(img)
            # draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            if label_scale_num['positive'] > 0:  # 随机偏移中心点以及生成新的边长
                new_side = side_ + side_ * np.random.uniform(-0.2, 0.2) + 1
                new_cx = cx + cx * np.random.uniform(-0.2, 0.2) + 1
                new_cy = cy + cy * np.random.uniform(-0.2, 0.2) + 1
            elif label_scale_num['part'] > 0:
                new_side = side_ + side_ * np.random.uniform(-1, 1) + 1
                new_cx = cx + cx * np.random.uniform(-1, 1) + 1
                new_cy = cy + cy * np.random.uniform(-1, 1) + 1
            '获得偏移框的坐标'
            x1_ = new_cx - new_side / 2
            y1_ = new_cy - new_side / 2
            x2_ = x1_ + new_side
            y2_ = y1_ + new_side

            # side_ = h_ = int(min(w, h) * np.random.uniform(0.8, 1.2))
            #
            # x_ = np.random.randint(int(x1 - w * 0.1), int(x1 + w * 0.1))
            # y_ = np.random.randint(int(y1 - h * 0.1), int(y1 + h * 0.1))
            # if x_ + side_ > img_w:
            #     x_ = img_w - side_ - 10
            # if x_ < 0:
            #     x_ = 10
            # if y_ + h_ > img_h:
            #     y_ = img_w - h_ - 10
            # if y_ < 0:
            #     y_ = 10
            # x1_, y1_, x2_, y2_ = x_, y_, x_ + side_, y_ + side_

            crop_box = np.array([x1_, y1_, x2_, y2_])
            offset_x1 = (x1 - x1_) / side_
            offset_y1 = (y1 - y1_) / side_
            offset_x2 = (x2 - x2_) / side_
            offset_y2 = (y2 - y2_) / side_
            iou_p = Iou(box_label, crop_box)

            # offset_px1 = (fx1 - x1_) / side_
            # offset_px2 = (fx2 - x1_) / side_
            # offset_px3 = (fx3 - x1_) / side_
            # offset_px4 = (fx4 - x1_) / side_
            # offset_px5 = (fx5 - x1_) / side_
            #
            # offset_py1 = (fy1 - y1_) / side_
            # offset_py2 = (fy2 - y1_) / side_
            # offset_py3 = (fy3 - y1_) / side_
            # offset_py4 = (fy4 - y1_) / side_
            # offset_py5 = (fy5 - y1_) / side_

            '(2)主要用于生成负样本'
            n_w = np.random.randint(int(min(img_w, img_h) * 0.30),
                                    int(min(img_w, img_h) * 0.39))
            nx1_ = np.random.randint(0, img_w - n_w)
            ny1_ = np.random.randint(0, img_h - n_w)
            nx2_ = nx1_ + n_w
            ny2_ = ny1_ + n_w

            crop_box_n = np.array([nx1_, ny1_, nx2_, ny2_])
            iou_n = Iou(box_label, crop_box_n)

            if iou_p >= 0.7 and label_scale_num['positive'] > 0:
                # draw.rectangle((x1_, y1_, x2_, y2_), outline="red", width=3)
                face_crop = img.crop(crop_box)
                face_resize = face_crop.resize((face_size, face_size))
                # if face_size == 48:
                #     positive_anno_file.write(
                #         f"positive/{positive_count}.jpg {1} {offset_x1} {offset_y1} {offset_x2} {offset_y2} "
                #         f"{offset_px1} {offset_py1} {offset_px2} {offset_py2} "
                #         f"{offset_px3} {offset_py3} {offset_px4} {offset_py4} "
                #         f"{offset_px5} {offset_py5}\n")
                #     positive_anno_file.flush()
                # else:
                positive_anno_file.write(
                    f"positive/{positive_count}.jpg {1} {offset_x1} {offset_y1} {offset_x2} {offset_y2}\n")
                positive_anno_file.flush()
                if label_scale_num['positive'] % 3 == 0:
                    # 随机增强
                    face_resize = enhance.rand_choice_enhance(face_resize, face_size, np.random.randint(0, 6))
                face_resize.save(f"{positive_image_dir}/{positive_count}.jpg")
                label_scale_num['positive'] -= 1
                positive_count += 1
            elif 0.3 < iou_p < 0.65 and label_scale_num['part'] > 0:
                # draw.rectangle((x1_, y1_, x2_, y2_), outline="blue", width=3)
                face_crop = img.crop(crop_box)
                face_resize = face_crop.resize((face_size, face_size))
                # if face_size == 48:
                #     part_anno_file.write(
                #         f"part/{part_count}.jpg {2} {offset_x1} {offset_y1} {offset_x2} {offset_y2} "
                #         f"{offset_px1} {offset_py1} {offset_px2} {offset_py2} "
                #         f"{offset_px3} {offset_py3} {offset_px4} {offset_py4} "
                #         f"{offset_px5} {offset_py5}\n")
                #     part_anno_file.flush()
                # else:
                part_anno_file.write(
                    f"part/{part_count}.jpg {2} {offset_x1} {offset_y1} {offset_x2} {offset_y2}\n")
                part_anno_file.flush()
                if label_scale_num['part'] % 3 == 0:
                    # 随机增强
                    face_resize = enhance.rand_choice_enhance(face_resize, face_size, np.random.randint(0, 6))
                face_resize.save(f"{part_image_dir}/{part_count}.jpg")
                label_scale_num['part'] -= 1
                part_count += 1
            elif iou_n < 0.05 and label_scale_num['negative'] > 0:
                # draw.rectangle((nx1_, ny1_, nx2_, ny2_), outline="black", width=3)
                face_crop = img.crop(crop_box_n)
                face_resize = face_crop.resize((face_size, face_size))
                # face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                negative_anno_file.write(
                    f"negative/{negative_count}.jpg {0} {0} {0} {0} {0} {0} {0} {0} {0} {0} {0} {0} {0} {0} {0}\n")
                negative_anno_file.flush()
                face_resize.save(f"{negative_image_dir}/{negative_count}.jpg")
                label_scale_num['negative'] -= 1
                negative_count += 1
            t2 = time.time()
            if label_scale_num['positive'] == label_scale_num['part'] == label_scale_num['negative'] == 0 or (
                    t2 - t1) > 1:
                break

            # plt.imshow(img)
            # # plt.show()
            # plt.axis("off")
            # plt.pause(0.1)
            # plt.clf()
    # print("over")
    positive_anno_file.close()
    part_anno_file.close()
    negative_anno_file.close()


if __name__ == '__main__':
    choices = {0: "train", 1: "val"}
    # choices = {0: "train"}
    label_position = r"../label\list_bbox_celeba.txt"
    # label_landmark_path = r"../label\list_landmarks_align_celeba.txt"
    img_path = r"F:\2.Dataset\img_celeba"
    start = 0
    end = 30000
    # scale_num = {"train": 0.7, "val": 0.15, "test": 0.15}
    scale_num = {"train": 0.8, "val": 0.05, "test": 0.19}

    position_label = get_sample(label_position)[start:end]  # 读标签
    # landmark_label = get_landmark(label_landmark_path)[:num]  # 读标签
    # position_label = get_sample(label_position)  # 读标签
    # landmark_label = get_landmark(label_landmark_path)  # 读标签
    # print(landmark_label)

    train_num = int(position_label.shape[0] * scale_num['train'])
    val_num = int(position_label.shape[0] * scale_num['val'])
    test_num = int(position_label.shape[0] - train_num - val_num)
    # train_num = int(len(position_label) * scale_num['train'])
    # val_num = int(len(position_label) * scale_num['val'])
    # test_num = int(len(position_label) - train_num - val_num)
    print(train_num, val_num, test_num)
    print(train_num * 15, val_num * 15, test_num)

    # position_label_split = [position_label[:train_num],
    #                         position_label[train_num:train_num + val_num],
    #                         position_label[train_num + val_num:]]
    # landmark_label_split = [landmark_label[:train_num],
    #                         landmark_label[train_num:train_num + val_num],
    #                         landmark_label[train_num + val_num:]]
    position_label_split = [position_label[:train_num],
                            position_label[train_num:train_num + val_num],
                            ]
    # landmark_label_split = [landmark_label[:train_num],
    #                         landmark_label[train_num:train_num + val_num],
    #                         ]

    t1 = time.time()
    pool = multiprocessing.Pool(10)
    for index, split in enumerate(position_label_split):
        # save_path = fr"F:\2.Dataset\mtcnn_12_m\{choices[index]}"
        save_path = fr"F:\Dataset02\mtcnn_fxs\{choices[index]}"
        print(save_path)
        # landmark = landmark_label_split[index]
        landmark = None
        # print(landmark)

        # for sample_size in [12, 48]:
        for sample_size in [12, 24, 48]:
            print(sample_size)
            # pool.apply_async(generator_sample, args=(split, landmark, img_path, save_path, sample_size))
            pool.apply_async(generator_sample, args=(split, landmark, img_path, save_path, sample_size))
            # generator_sample(split, landmark, img_path, save_path, sample_size)
    pool.close()
    pool.join()
    print(time.time() - t1)  # 9.64033031463623~11.881264686584473  可以进行提速
    print("Ending...")
