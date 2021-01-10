import time
import torch
from PIL import Image, ImageDraw
import os
from detector import Detector, show_single_image

if __name__ == '__main__':
    x = time.time()
    with torch.no_grad() as grad:
        image_file = r"images2\9.jpg"
        # image_file = r"F:\2.Dataset\img_celeba/000001.jpg"
        image_path = r"images2"
        show_single_image(image_file)
        exit()
        for i in os.listdir(image_path):
            with Image.open(os.path.join(image_path, i)) as im:
                detector = Detector()
                # p_net_param=r".\params_p/p_net_0_s.pth",
                #                     r_net_param="./params_r/r_net_13_s.pth",
                #                     o_net_param="./params_o/o_net_45.pth")
                w, h = im.size
                img = im.resize((w // 2, h // 2))
                boxes = detector.detect(img)
                # print(boxes.shape)
                imDraw = ImageDraw.Draw(im)
                for box in boxes:
                    x1 = int(box[0] * 2)
                    y1 = int(box[1] * 2)
                    x2 = int(box[2] * 2)
                    y2 = int(box[3] * 2)
                    # print(box[4])
                    imDraw.rectangle((x1, y1, x2, y2), outline='red', width=3)
                y = time.time()
                # print(y - x)
                im.show()
