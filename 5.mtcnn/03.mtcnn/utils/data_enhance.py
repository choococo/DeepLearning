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
# img = Image.open(r"image/000120.jpg")


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
    # def salt_and_pepper_noise(img, proportion=0.05):
    #     def ranColor():
    #         return (np.random.randint(56, 255),
    #                 np.random.randint(56, 255),
    #                 np.random.randint(56, 255))
    #
    #     # print(noise_img.shape)
    #     draw = ImageDraw.Draw(img)
    #     noise_img = np.array(img)
    #     height, width, channel = noise_img.shape[0], noise_img.shape[1], noise_img[2]
    #     num = int(height * width * proportion)  # 多少个像素点添加椒盐噪声
    #     for i in range(num):
    #         w = np.random.randint(0, width - 1)
    #         h = np.random.randint(0, height - 1)
    #         if np.random.randint(0, 1) == 0:
    #             draw.point((w, h), ranColor())
    #             # noise_img[h, w, 0] = np.random.randint(0, 255)
    #             # noise_img[h, w, 1] = np.random.randint(0, 255)
    #             # noise_img[h, w, 2] = np.random.randint(0, 255)
    #         else:
    #             draw.point((w, h), ranColor())
    #             # noise_img[h, w] = np.random.randint(0, 255)
    #     # return Image.fromarray(noise_img)
    #     return img

    # 4. 加入噪声
    # img_noise = salt_and_pepper_noise(img)
    # return img_noise
    img_noise = random_noise(np.array(img), 'poisson')
    img_noise = np.uint8(img_noise * 255)
    img_noise = Image.fromarray(img_noise)
    return img_noise


def enh_gaussian_blur(img):
    # 3. 加入高斯模糊
    factor_gaussian_blur = np.random.uniform(0.6, 0.9)
    img_filter = img.filter(ImageFilter.GaussianBlur(factor_gaussian_blur))
    return img_filter


def rand_choice_enhance(image, flag=0):
    name = ["enh_brightness_dark", "enh_noise", "enh_gaussian_blur"]
    # 高内聚，低耦合，对if-else进行优化
    if flag == 0:
        return enh_brightness_dark(image)
        # return eval(name[0])(image)
        # return enh_brightness_dark(image)
    elif flag == 1:
        return enh_noise(image)
    elif flag == 2:
        return enh_gaussian_blur(image)
    elif flag == 3:
        image = enh_noise(image)
        image = enh_brightness_dark(image)
        return image
    elif flag == 4:
        image = enh_gaussian_blur(image)
        image = enh_noise(image)
        return image
    elif flag == 5:
        image = enh_gaussian_blur(image)
        image = enh_noise(image)
        image = enh_brightness_dark(image)
        return image
    # elif flag == 7:
    #     image = enh_brightness_dark(image)
    #     image = enh_gaussian_blur(image)
    #     image = enh_noise(image)
    #     return image


# img.show()
# img_noise = rand_choice_enhance(img, 0).show()


# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.axis("off")
#
# plt.subplot(1, 2, 2)
# plt.imshow(img_noise)
# plt.axis("off")
# plt.show()

