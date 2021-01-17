import os
import time
from PIL import Image
import numpy as np
from utils.tools import iou
import random
from data_enhance import rand_choice_enhance as img_enhance

'源标签和图片的路径'
original_label_dir = r"C:\Users\liewei\Desktop\03.mtcnn\label\list_bbox_celeba.txt"
img_dir = r"F:\2.Dataset\img_celeba"


def get_sample(save_path, sample_size, stop_value):
    pst_img_dir = os.path.join(save_path, str(sample_size), "positive")
    part_img_dir = os.path.join(save_path, str(sample_size), "part")
    ngt_img_dir = os.path.join(save_path, str(sample_size), "negative")
    '创建保存不同尺寸图片的文件路径'
    for dir_path in [pst_img_dir, part_img_dir, ngt_img_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    '创建保存样本图片的标签文件路径'
    pst_label_filename = os.path.join(save_path, str(sample_size), 'positive.txt')
    part_label_filename = os.path.join(save_path, str(sample_size), 'part.txt')
    ngt_label_filename = os.path.join(save_path, str(sample_size), 'negative.txt')
    '样本的统计数'
    pst_count = 1
    part_count = 1
    ngt_count = 1

    pst_label_file = open(pst_label_filename, 'w')  # 将保存标签的空文件打开并写入标签
    part_label_file = open(part_label_filename, 'w')
    ngt_label_file = open(ngt_label_filename, 'w')
    original_label = open(original_label_dir, 'r')
    # with open(original_label_dir, 'r') as f:
    # for i, line in enumerate(open(original_label_dir,'r')):  # 遍历出储存原始标签的文件，i是给每行标签给的一个索引
    #     if i < 2:
    #         continue  # 和数据无关的前两行直接跳过
    # for line in original_label.readlines()[171000:200000]:
    for line in original_label.readlines()[0:100]:
        flag = 0
        strs = line.strip().split()
        print(strs)
        img_name = strs[0]
        image_file = os.path.join(img_dir, img_name)  # 每张图片的绝对路径
        with Image.open(image_file) as img:  # 打开图片
            img_w, img_h = img.size
            x1 = float(strs[1])
            y1 = float(strs[2])
            w = float(strs[3])
            h = float(strs[4])
            if x1 < 0 or y1 < 0 or w < 0 or h < 0 or (w * h) / (max(w, h) * max(w, h)) <= 0.7:  # 去除不合格的样本标签
                continue  # or max(w,h)<40 or max(w,h)>400
            x2 = x1 + w
            y2 = y1 + h

            boxes = [[x1, y1, x2, y2]]  # 将原始框的坐标标签放进二维列表中，后面对其进行一步计算IOU
            '原中心点'
            cx = x1 + w / 2
            cy = y1 + h / 2
            '随机选择一个边长，后面对其进行缩放'
            side_len = np.random.choice([w, h])
            '每张图片生成三种样本的数量'
            number = [3, 3, 9]
            while True:
                if flag > 1000:  # 如果循环800次还没有找到符合条件的iou截图，就结束这个循环，该图片直接舍弃
                    break
                flag += 1
                if number[0] > 0:  # 随机偏移中心点以及生成新的边长
                    new_side = side_len + side_len * np.random.uniform(-0.2, 0.2) + 1
                    new_cx = cx + cx * random.uniform(-0.2, 0.2) + 1
                    new_cy = cy + cy * random.uniform(-0.2, 0.2) + 1
                elif number[1] > 0:
                    new_side = side_len + side_len * np.random.uniform(-1, 1) + 1
                    new_cx = cx + cx * random.uniform(-1, 1) + 1
                    new_cy = cy + cy * random.uniform(-1, 1) + 1
                elif number[2] > 0:
                    new_side = side_len + side_len * np.random.uniform(-3, 3)
                    new_cx = cx + cx * random.uniform(-3, 3) + 1
                    new_cy = cy + cy * random.uniform(-3, 3) + 1

                '获得偏移框的坐标'
                x1_ = new_cx - new_side / 2
                y1_ = new_cy - new_side / 2
                x2_ = x1_ + new_side
                y2_ = y1_ + new_side

                '判断新中心点的位置，确保新的偏移框不会框到图外面'
                if x1_ < 0 or y1_ < 0 or x2_ > img_w or y2_ > img_h or new_side < sample_size or \
                        new_side / 2 > new_cx > img_w - new_side / 2 or new_side / 2 > new_cy > img_h - new_side / 2:
                    continue
                    # 判断偏移超出整张图片的就跳过，不截图

                '获得偏移量'
                offset_x1 = (x1 - x1_) / new_side
                offset_y1 = (y1 - y1_) / new_side
                offset_x2 = (x2 - x2_) / new_side
                offset_y2 = (y2 - y2_) / new_side
                '裁剪偏移框内的图像并缩放'
                crop_box = [x1_, y1_, x2_, y2_]  # 将(偏移框)需要裁剪的坐标放进列表中，下面会让其与原始框坐标进行统一的IOU计算
                img_crop = img.crop(crop_box)
                img_resize = img_crop.resize((sample_size, sample_size))

                '计算iou'
                Iou = iou(crop_box, np.array(boxes))[0]  # 因为裁剪的坐标和原始框的坐标值都放在列表中，所以计算出来的iou也是
                # 放在列表中的，因此需要将这些放在一维列表中的iou值索引出来

                '保存图片和标签'
                if Iou >= 0.7 and number[0] > 0:
                    pst_label_file.write(
                        'positive/{0}.jpg {1} {2} {3} {4} {5}\n'.format(pst_count, 1,
                                                                        offset_x1, offset_y1, offset_x2,
                                                                        offset_y2))
                    pst_label_file.flush()
                    if number[0] == 1:
                        img_resize = img_enhance(img_resize, sample_size,np.random.randint(0, 7))  # 随机选择一种噪声加到正样本中的一张图片上
                    img_resize.save(os.path.join(pst_img_dir, "{0}.jpg".format(pst_count)))
                    number[0] -= 1
                    pst_count += 1
                    print(1)
                elif 0.3 <= Iou < 0.7 and number[1] > 0:
                    part_label_file.write(
                        "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2,
                                                                    offset_x1, offset_y1, offset_x2, offset_y2))
                    part_label_file.flush()
                    if number[1] == 2:
                        img_resize = img_enhance(img_resize,sample_size, np.random.randint(0, 7))
                    img_resize.save(os.path.join(part_img_dir, "{0}.jpg".format(part_count)))
                    number[1] -= 1
                    part_count += 1
                    print(2)
                elif Iou < 0.3 and number[2] > 0:
                    ngt_label_file.write("negative/{0}.jpg {1} {2} {3} {4} {5}\n".format(ngt_count, 0,
                                                                                         offset_x1, offset_y1,
                                                                                         offset_x2, offset_y2))
                    ngt_label_file.flush()
                    img_resize.save(os.path.join(ngt_img_dir, "{0}.jpg".format(ngt_count)))
                    number[2] -= 1
                    ngt_count += 1
                    print(0)

                if number[0] == 0 and number[1] == 0 and number[2] == 0:  # 样本生成时间大于1秒，直接下一次循环
                    break
        count = pst_count + part_count + ngt_count  # 计算生成样本的总数，达到设定值就停止生成
        if count >= stop_value:
            break


if __name__ == '__main__':
    import multiprocessing
    pool = multiprocessing.Pool(10)
    save_path = r"F:\Dataset02\mtcnn_fxs"
    # get_sample(save_path, 12, 240000)
    get_sample(save_path, 24, 150)
    # get_sample(save_path, 48, 240000)
    # for face_size in [24, 48]:
    #
    #     pool.apply_async(get_sample, args=(save_path, face_size, 240000))
    #     # get_sample(save_path, 24, 240000)
    # pool.close()
    # pool.join()