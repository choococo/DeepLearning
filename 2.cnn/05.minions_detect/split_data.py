import numpy as np
import random
import os
import sys
import shutil
from PIL import Image, ImageDraw

"""
划分数据集：训练集、验证集、测试集
数据的类型，所有做好标签的数据都在一个文件夹中，部分代码可以适用于一遍生成样本，一遍分类
步骤：
    首先要生成三个文件夹：train、val、test
    标签文件：train_label.txt val_label.txt test_label.txt
"""


def split_bg_data_3():
    """
    将背景图拆分成Train、Val、Test
    :return:
    """
    choices = {0: "train", 1: "val", 2: "test"}
    origin_path = r"F:\2.Dataset\Yellow\bg\background"
    # bg_linshi_path = r"F:\2.Dataset\Yellow\bg\bg_split"  # 7 1.5 1.5
    bg_linshi_path = r"F:\2.Dataset\Yellow\Minions_Test\bg"  # 7 1.5 1.5

    # 1.
    bg_img_list = []
    for i, bg_name in enumerate(os.listdir(origin_path)):
        # 拆分数据集
        bg_img_list.append(bg_name)
    print(len(bg_img_list))

    # res1 = random.sample(bg_img_list, int(len(bg_img_list) * 0.7))
    res1 = random.sample(bg_img_list, 1000)
    print(len(res1))
    for img_name in res1:
        shutil.move(os.path.join(origin_path, img_name), os.path.join(bg_linshi_path, choices[0], img_name))

    # 2.
    bg_img_list1 = []
    for i, bg_name in enumerate(os.listdir(origin_path)):
        # 拆分数据集
        bg_img_list1.append(bg_name)
    print(len(bg_img_list1))

    # res2 = random.sample(bg_img_list1, int(len(bg_img_list1) * 0.5))
    res2 = random.sample(bg_img_list1, 1000)
    print(bg_img_list1)
    for img_name in res2:
        shutil.move(os.path.join(origin_path, img_name), os.path.join(bg_linshi_path, choices[1], img_name))

    # 3.
    bg_img_list2 = []
    for i, bg_name in enumerate(os.listdir(origin_path)):
        # 拆分数据集
        bg_img_list2.append(bg_name)
    # res3 = random.sample(bg_img_list2, int(len(bg_img_list2)))
    res3 = random.sample(bg_img_list2, 1000)
    print(res3)
    for img_name in res3:
        shutil.move(os.path.join(origin_path, img_name), os.path.join(bg_linshi_path, choices[2], img_name))


def add_positive_negative():
    """
    将背景贴上小黄人，制作成positive和negative
    :return:
    """
    choices = {0: "train", 1: "val", 2: "test"}
    minions_path = r"F:\2.Dataset\Yellow\yellow"
    root = r"F:\2.Dataset\Yellow\Minions"
    for i in choices:
        if not os.path.exists(f"{root}/{choices[i]}"):
            os.mkdir(f"{root}/{choices[i]}")
        bg_path = fr"F:\2.Dataset\Yellow\bg\bg_split\{choices[i]}"
        # bg_split = fr"F:\2.Dataset\Yellow\bg\bg_split\{choices[0]}"
        save_path = f"{root}/{choices[i]}"
        label_path = fr"F:\2.Dataset\Yellow\Minions/{choices[i]}_label.txt"
        length = len(os.listdir(bg_path)) * len(os.listdir(minions_path))
        with open(label_path, "w") as f:
            count = 0
            for bg_name in os.listdir(bg_path):
                # 1. 小黄人
                num = random.randint(1, 20)
                # print(num)
                image = Image.open(f"{minions_path}/{num}.png")

                # new_len = random.randint(100, 130)
                # image_resize = image.resize((new_len, new_len))
                # new_w, new_h = random.randint(100, 128), random.randint(100, 128)
                new_w, new_h = random.randint(60, 128), random.randint(60, 128)
                image_resize = image.resize((new_w, new_h))

                image_roate = image_resize.rotate(random.randint(-45, 45))  # 旋转
                # image_roate.show()
                # 2.背景图，先进行224的缩放,然后贴小黄人，这样坐标就统一了
                img_bg = Image.open(f"{bg_path}/{bg_name}")
                img_bg = img_bg.convert("RGBA")
                img_bg = img_bg.resize((224, 224))
                img_bg.save(f"{save_path}/{count}_bg.png")  # 将图片进行保存
                f.write(f"{count}_bg.png {0} {0} {0} {0} {0}\n")
                f.flush()
                # r, g, b, a = image_roate.split() # 路径拆分
                # 3. 生成随机粘贴坐标
                paste_x, paste_y = random.randint(0, 224 - new_w), random.randint(0, 224 - new_h)
                # paste_x, paste_y = random.randint(0, 224 - new_len), random.randint(0, 224 - new_len)
                # print(paste_x, paste_y)

                img_bg.paste(image_roate, (paste_x, paste_y), mask=image_roate)  # 把透明通道粘贴上
                # paste_x2 = paste_x + new_len
                # paste_y2 = paste_y + new_len
                paste_x2 = paste_x + new_w
                paste_y2 = paste_y + new_h
                img_bg.save(f"{save_path}/{count}.png")
                f.write(f"{count}.png {1} {paste_x} {paste_y} {paste_x2} {paste_y2}\n")
                f.flush()
                count += 1
                # 加入进度条
                sys.stdout.write("\r >> processing {}/{}".format((count), length))
                sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()


if __name__ == '__main__':
    # 拆分背景图成Train、Val、Test
    # split_bg_data_3()
    # pass
    add_positive_negative()
