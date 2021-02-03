import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import skimage
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import random
from utils.tools import *
import traceback

boxs_path = r'../CelebA/Anno/list_bbox_celeba.txt'
img_path = r'../CelebA/Img/img_celeba'


class Sample():
    def __init__(self, face_size, stop_num, save_path):

        self.count = 0

        '''创建存放图片的文件夹'''
        img_name = ['positive', 'negative', 'part']
        for name in img_name:
            if not os.path.exists(os.path.join(save_path, str(face_size), name)):
                os.makedirs(os.path.join(save_path, str(face_size), name))
        try:

            '''创建txt并打开'''
            f_positive = open(os.path.join(save_path, str(face_size), 'positive.txt', ), 'w', encoding='utf-8')
            f_negative = open(os.path.join(save_path, str(face_size), 'negative.txt'), 'w', encoding='utf-8')
            f_part = open(os.path.join(save_path, str(face_size), 'part.txt'), 'w', encoding='utf-8')

            for i, line in enumerate(open(boxs_path)):

                try:

                    path, x1, y1, w, h = line.split()
                    x1, y1, w, h = float(x1), float(y1), float(w), float(h)
                    print(path)

                    '''去掉数据中无用的数据'''
                    if x1 <= 0 or y1 <= 0 or w <= 0 or h <= 0 or ((w * h) / (max(w, h) * max(w, h)) <= 0.7):
                        continue

                    '''中心点、边长、x2,y2'''
                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    x2 = x1 + w
                    y2 = y1 + h
                    side_len = random.choice((w, h))

                    positive_num = 0
                    negative_num = 0
                    part_num = 0

                    '''将图片打开'''
                    img = Image.open(os.path.join(img_path, path))
                    img_w, img_h = img.size
                    img_num = 0  # 对该图片遍历的次数

                    '''对每一张图片进行遍历'''
                    while True:

                        '''防止特殊图片出现'''
                        if img_num > 5000:
                            break
                        img_num += 1  # 对该图片循环的次数

                        if positive_num < 3:
                            _side_len = np.maximum(side_len + side_len * random.uniform(-0.2, 0.2) + 1, face_size)
                            _cx = cx + cx * random.uniform(-0.2, 0.2) + 1
                            _cy = cy + cy * random.uniform(-0.2, 0.2) + 1
                        elif part_num < 3:
                            _side_len = np.maximum(side_len + side_len * random.uniform(-1, 1) + 1, face_size)
                            _cx = cx + cx * random.uniform(-1, 1) + 1
                            _cy = cy + cy * random.uniform(-1, 1) + 1
                        elif negative_num < 9:
                            _side_len = np.random.uniform(face_size, np.minimum(h, w) / 2)
                            x_list = [
                                [_side_len / 2, x1, _side_len / 2, img_h - _side_len / 2],
                                [x1, x2, _side_len / 2, y1],
                                [x1, x2, y2, img_h - _side_len / 2],
                                [x2, img_w - _side_len / 2, _side_len / 2, img_h - _side_len / 2]
                            ]
                            a = random.choice(x_list)
                            _cx = np.random.uniform(a[0], a[1])
                            _cy = np.random.uniform(a[2], a[3])

                            # _cx=np.random.uniform(_side_len/2,img_w-_side_len/2)
                            # _cy = np.random.uniform(_side_len/2,img_h-_side_len/2)

                        _x1 = np.maximum(_cx - _side_len / 2, 0)
                        _y1 = np.maximum(_cy - _side_len / 2, 0)
                        _x2 = np.minimum(_x1 + _side_len, img_w)
                        _y2 = np.minimum(_y1 + _side_len, img_h)

                        '''边界检查'''
                        # if _side_len<face_size or _x1<0 or _y1<0 or _x2>img_w or _y2>img_h:
                        #     continue

                        '''计算IOU值'''
                        iou_ratio = iou([x1, y1, x2, y2], np.array([[_x1, _y1, _x2, _y2]]))[0]
                        print(iou_ratio)

                        '''偏移量'''
                        offset_x1 = (x1 - _x1) / _side_len
                        offset_y1 = (y1 - _y1) / _side_len
                        offset_x2 = (x2 - _x2) / _side_len
                        offset_y2 = (y2 - _y2) / _side_len

                        '''裁剪图片'''
                        image = img.crop((_x1, _y1, _x2, _y2))
                        image = image.resize((face_size, face_size), Image.ANTIALIAS)
                        image = self.img_enhance(image)

                        '''标签制作和图片保存'''
                        if iou_ratio > 0.6 and positive_num < 3:
                            f_positive.write(
                                'positive/{}_{}.jpg,1,{},{},{},{}\n'.format(i, positive_num, offset_x1, offset_y1,
                                                                            offset_x2, offset_y2))
                            f_positive.flush()
                            image.save(os.path.join(save_path, str(face_size), img_name[0],
                                                    '{}_{}.jpg'.format(i, positive_num)))
                            image.show()
                            exit()
                            positive_num += 1
                            self.count += 1
                            print(self.count, 'positive')

                        elif 0.5 > iou_ratio > 0.3 and part_num < 3:
                            f_part.write(
                                'part/{}_{}.jpg,2,{},{},{},{}\n'.format(i, part_num, offset_x1, offset_y1, offset_x2,
                                                                        offset_y2))
                            f_part.flush()
                            image.save(
                                os.path.join(save_path, str(face_size), img_name[2], '{}_{}.jpg'.format(i, part_num)))
                            part_num += 1
                            self.count += 1
                            print(self.count, 'part')


                        elif (iou_ratio < 0.2 or iou_ratio == 0) and negative_num < 9:
                            f_negative.write('negative/{}_{}.jpg,0,0,0,0,0\n'.format(i, negative_num))
                            f_negative.flush()
                            image.save(os.path.join(save_path, str(face_size), img_name[1],
                                                    '{}_{}.jpg'.format(i, negative_num)))
                            negative_num += 1
                            self.count += 1
                            print(self.count, 'negative')

                        '''一张图片==》15，循环下一张图片'''
                        if positive_num + negative_num + part_num >= 15:
                            break

                    if self.count >= stop_num:
                        break

                except:
                    traceback.print_exc()
        except:
            traceback.print_exc()

    '''图片添加噪声'''

    def img_enhance(self, img):
        if np.random.random() < 0.4:  # 模糊
            img = img.filter(ImageFilter.GaussianBlur(np.random.randint(1, 3)))
        if np.random.random() < 0.4:  # 噪声
            img = skimage.util.random_noise(np.array(img), mode='gaussian')
            img = np.uint8(img * 255)
            img = Image.fromarray(img)
        if np.random.random() < 0.4:  # 变暗
            img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.5, 1))

        return img


if __name__ == '__main__':
    train_path = r'./TRAIN3'
    verify_path = r'./VERIFY3'
    sample = Sample(48, 1, train_path)
    # sample1=Sample(12,1,train_path)
    # sample2=Sample(24,1,train_path)
    # sample3=Sample(48,10000,verify_path)
    # sample4=Sample(12,10000,verify_path)
    # sample5=Sample(24,10000,verify_path)
