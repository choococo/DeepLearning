import time
import torch
from PIL import Image, ImageDraw
import os
from detector_lzy import Detector, show_single_image

if __name__ == '__main__':
    x = time.time()
    with torch.no_grad() as grad:
        image_file = r"images\10.jpg"
        image_path = r"images2"
        # show_single_image(image_file)
        # exit()
        for i in os.listdir(image_path):
            with Image.open(os.path.join(image_path, i)) as im:
                detector = Detector()
                # w, h = im.size
                # img = im.resize((w // 2, h // 2))
                boxes = detector.detect(im)
                # print(boxes.shape)
                imDraw = ImageDraw.Draw(im)
                for box in boxes:
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])
                    # print(box[4])
                    imDraw.rectangle((x1, y1, x2, y2), outline='red', width=3)
                y = time.time()
                # print(y - x)
                im.show()
