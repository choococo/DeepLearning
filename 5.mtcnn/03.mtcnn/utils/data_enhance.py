import numpy as np
import os
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from skimage.util import random_noise
import matplotlib.pyplot as plt

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
    enh_brightness = ImageEnhance.Brightness(img)
    factor_dark = np.random.uniform(0.4, 0.6)
    factor_bright = np.random.uniform(1, 1.3)
    factor = np.random.choice([factor_dark, factor_bright])
    img_brightness = enh_brightness.enhance(factor=factor)
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
    img_noise = random_noise(np.array(img), 'poisson')
    img_noise = np.uint8(img_noise * 255)
    img_noise = Image.fromarray(img_noise)
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
    # elif flag == 7:
    #     image = enh_brightness_dark(image)
    #     image = enh_gaussian_blur(image)
    #     image = enh_noise(image)
    #     return image


if __name__ == '__main__':
    img = Image.open(r"image/000120.jpg")
    img.show()
    img_noise = rand_choice_enhance(img, 0).show()

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_noise)
    plt.axis("off")
    plt.show()
