from PIL import Image
from PIL import ImageDraw
import os

img = Image.open("")
imgDraw = ImageDraw.Draw(img)
# 70   8  68 119
imgDraw.rectangle((70, 8, 70 + 68, 8 + 119), outline="red")
img.show()
