from nets import *
import torch
from torchvision import transforms
import cv2
from utils.tools import *
from PIL import Image, ImageFilter, ImageDraw
import time


class Detector:

    def __init__(self):
        self.pnet = PNet()
        self.pnet.load_state_dict(torch.load('./save_params/p_net.pth'))

        self.rnet = RNet()
        self.rnet.load_state_dict(torch.load('./save_params/r_net.pth'))

        self.onet = ONet()
        self.onet.load_state_dict(torch.load('./save_params/o_net.pth'))

        '''保持参数稳定'''
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect(self, img):
        '''1、将图片放入Pnet网络中==>得到一组框'''
        p_boxes = self.detPnet(img)
        if p_boxes is None:
            return []

        # boxes = convert_to_square(p_boxes)
        # r_boxes = self.detRnet(img, boxes)
        # if r_boxes is None:
        #     return []
        #
        # boxes = convert_to_square(r_boxes)
        # o_boxes = self.detOnet(img, boxes)
        # if o_boxes is None:
        #     return []

        return p_boxes

    def detPnet(self, img):
        w, h = img.size

        scale = 1  # 对图片按照该比例进行缩放
        scale_img = img
        min_side = min((w, h))
        _boxes = []
        while min_side >= 12:
            img_data = self.tf(scale_img)  # 图片转成tensor
            img_data = img_data.unsqueeze(0)  # 给传入的图片增加一个维度
            cond, offset = self.pnet(img_data)

            '''取出可能有人的索引'''
            # 原图卷积后，每个1x1对应在原图上的12x12
            _cond = cond[0, 0]  # _cond:torch.Size([814, 1075])
            c_mask = _cond > 0.6  # 大于阈值则可能有人，找出可能为人的点
            '''获取有人的置信度的'''
            cc = _cond[c_mask]
            idx = c_mask.nonzero()

            _x1, _y1 = (idx[:, 1] * 2) / scale, (idx[:, 0] * 2) / scale  # 2为整个P网络代表的步长
            _x2, _y2 = (idx[:, 1] * 2 + 12) / scale,(idx[:, 0] * 2 + 12) / scale

            _w = _x2 - _x1
            _h = _y2 - _y1

            '''获取原图中可能有坐标的点'''
            p = offset[0, :, c_mask]
            x1 = (p[0, :] * _w + _x1)
            y1 = (p[1, :] * _h + _y1)
            x2 = (p[2, :] * _w + _x2)
            y2 = (p[3, :] * _h + _y2)

            '''拼接'''
            _boxes.append(torch.stack((cc, x1, y1, x2, y2), dim=1))

            '''图像金字塔'''
            scale *= 0.702
            w, h = int(w * scale), int(h * scale)

            scale_img = scale_img.resize((w, h), Image.ANTIALIAS)
            min_side = min(w, h)

        '''所有标签值拼接成【n,5】的形状'''
        boxes = torch.cat(_boxes, dim=0)

        return nms(np.array(_boxes), 0.6)

    def detRnet(self, img, boxes):
        boxes = self.__rnet_onet(img, 24, boxes)
        return Nms(boxes, 0.4)

    def detOnet(self, img, boxes):
        boxes = self.__rnet_onet(img, 48, boxes)
        return nms(boxes, 0.4, isMin=True)

    def __rnet_onet(self, img, size, boxes):
        imgs = []
        for box in boxes:
            _x1 = int(box[1])
            _y1 = int(box[2])
            _x2 = int(box[3])
            _y2 = int(box[4])
            _img = img.crop((_x1, _y1, _x2, _y2))
            resize_img = _img.resize((size, size),Image.ANTIALIAS)
            resize_img = self.tf(resize_img)
            imgs.append(resize_img)

        _imgs = torch.stack(imgs, dim=0)

        if size == 24:
            cond, offset = self.rnet(_imgs)
        else:
            cond, offset = self.onet(_imgs)

        '''
        boxes.shape:(3915, 5)
        cond.shape:torch.Size([3915, 1])
        offset.shape:torch.Size([3915, 4])
        '''

        c_mask = cond[:, 0] > 0.6  # c_mask: torch.Size([3915])
        _boxes = torch.tensor(boxes[c_mask])  # 得到置信度满足条件的框   #_boxes :(4, 5)
        cc = cond[c_mask].view(-1)  # cond:torch.Size([4,])
        _offset = offset[c_mask]  # _offset :torch.Size([4, 4])

        _w, _h = _boxes[:, 3] - _boxes[:, 1], _boxes[:, 4] - _boxes[:, 2]
        x1 = _offset[:, 0] * _w + _boxes[:, 1]
        y1 = _offset[:, 1] * _h + _boxes[:, 2]
        x2 = _offset[:, 2] * _w + _boxes[:, 3]
        y2 = _offset[:, 3] * _h + _boxes[:, 4]

        boxes = torch.stack((cc, x1, y1, x2, y2), dim=1)

        return boxes


if __name__ == '__main__':
    x = time.time()
    img_path = r'image/1.jpg'
    detector = Detector()
    with Image.open(img_path) as im:
        boxes = detector.detect(im)
        print(boxes.shape)
        imDraw = ImageDraw.Draw(im)
        for box in boxes:
            x1 = int(box[1])
            y1 = int(box[2])
            x2 = int(box[3])
            y2 = int(box[4])
            imDraw.rectangle((x1, y1, x2, y2), outline='red', width=3)
        y = time.time()
        print(y - x)
        im.show()
