from PIL import Image
from PIL import ImageDraw
import os

# IMG_DIR = r"E:\datasets\img_celeba_ALL"
# ANO_DIR = r"E:\datasets\list_bbox_celeba_ALL.txt"

img = Image.open("D:/celeba_1w/000120.jpg")
imgDraw = ImageDraw.Draw(img)
# 70   8  68 119
imgDraw.rectangle((70, 8, 70 + 68, 8 + 119), outline="red")
img.show()
