import numpy as np
import os
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from skimage.util import random_noise
import matplotlib.pyplot as plt
from torchvision import transforms
import Augmentor

"""
数据增强
    （1）变暗
    （2）变亮
    （3）加入噪声
    （4）加入高斯模糊
    （5）随机两个或三个或四个
"""


def enh_brightness_dark(img):
    # 1. 变暗
    enh_brightness = ImageEnhance.Brightness(img)  # 需要加入明暗度的图片
    factor_dark = np.random.uniform(0.4, 0.6)  # 暗度的因子
    factor_bright = np.random.uniform(1, 1.3)  # 明度因子
    factor = np.random.choice([factor_dark, factor_bright])  # 随机选取明度和暗度
    img_brightness = enh_brightness.enhance(factor=0.3)  # 进行明暗调节
    # img_brightness.show()
    return img_brightness


def enh_noise(img):
    # width, height = img.size
    # image = np.array(img)
    # num = int(height * width * 0.06)  # 多少个像素点添加椒盐噪声
    # for i in range(num):
    #     w = np.random.randint(0, width - 1)
    #     h = np.random.randint(0, height - 1)
    #     if np.random.randint(0, 1) == 0:
    #         image[h, w, 0] = np.random.randint(0, 255)
    #         image[h, w, 1] = np.random.randint(0, 255)
    #         image[h, w, 2] = np.random.randint(0, 255)
    #     else:
    #         image[h, w, 0] = np.random.randint(0, 255)
    #         image[h, w, 1] = np.random.randint(0, 255)
    #         image[h, w, 2] = np.random.randint(0, 255)
    # image = Image.fromarray(image)
    # return image
    img_noise = random_noise(np.array(img), 'poisson')  # poisson类型噪声，里面有很多
    img_noise = np.uint8(img_noise * 255)  # 上面对数据会自动归一化，因此这里需要转回图片
    img_noise = Image.fromarray(img_noise)  # 转回图片
    return img_noise


def enh_gaussian_blur(img, face_sie):
    # 3. 加入高斯模糊
    if face_sie == 48:
        factor_gaussian_blur = np.random.uniform(1.5, 2)
    elif face_sie == 24:
        factor_gaussian_blur = np.random.uniform(1, 1.5)
    else:
        factor_gaussian_blur = np.random.uniform(0.6, 1.1)
    img_filter = img.filter(ImageFilter.GaussianBlur(factor_gaussian_blur))
    return img_filter


def black_block(img, face_size):

    return img


def rand_choice_enhance(image, faces_size, flag=0):
    name = ["enh_brightness_dark", "enh_noise", "enh_gaussian_blur"]
    # 高内聚，低耦合，对if-else进行优化
    if flag == 0:
        return enh_brightness_dark(image)
        # return eval(name[0])(image)
        # return enh_brightness_dark(image)
    elif flag == 1:
        return enh_noise(image)
    elif flag == 2:
        return enh_gaussian_blur(image, faces_size)
    elif flag == 3:
        image = enh_brightness_dark(image)
        image = enh_noise(image)
        return image
    elif flag == 4:
        image = enh_noise(image)
        image = enh_gaussian_blur(image, faces_size)
        return image
    elif flag == 5:
        image = enh_brightness_dark(image)
        image = enh_noise(image)
        image = enh_gaussian_blur(image, faces_size)

        return image
    elif flag == 7:
        image = enh_brightness_dark(image)
        image = enh_gaussian_blur(image)
        image = enh_noise(image)
        return image


if __name__ == '__main__':
    img = Image.open(r"F:\2.Dataset\img_celeba/000001.jpg")
    # img.show()
    img_noise = rand_choice_enhance(img, 24, 0)
    img_noise.save("../img01.jpg")
    img_n = rand_choice_enhance(img, 24, np.random.randint(0, 7))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_noise)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img_n)
    plt.axis("off")

    plt.show()
