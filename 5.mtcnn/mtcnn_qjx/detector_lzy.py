import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
from utils import tools
import nets
import time
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Detector:
    def __init__(self, p_net_param=r"./save_params/p_net.pth",
                 r_net_param=r"./save_params/r_net.pth",
                 o_net_param="./save_params/o_net.pth",
                 isCuda=False):
        self.isCuda = isCuda
        self.p_net = nets.PNet().to(DEVICE)
        self.r_net = nets.RNet().to(DEVICE)
        self.o_net = nets.ONet().to(DEVICE)

        self.p_net.load_state_dict(torch.load(p_net_param))
        self.r_net.load_state_dict(torch.load(r_net_param))
        self.o_net.load_state_dict(torch.load(o_net_param))
        self.p_net.eval()
        self.r_net.eval()
        self.o_net.eval()
        self.__image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对于图像这个是有经验值的
        ])

    def detect(self, image):
        start_time = time.time()
        p_net_boxes = self.__p_net_detect(image)
        if p_net_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_p_net = end_time - start_time
        # return p_net_boxes

        start_time = time.time()
        r_net_boxes = self.__ro_net_detect(image, p_net_boxes, face_size=24)
        if r_net_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_r_net = end_time - start_time
        # return r_net_boxes

        start_time = time.time()
        o_net_boxes = self.__ro_net_detect(image, r_net_boxes, face_size=48)
        if o_net_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_o_net = end_time - start_time
        total_time = t_p_net + t_r_net + t_o_net
        print(f"[time]  total:{total_time} p_net:{t_p_net} r_net:{t_r_net} o_net:{t_o_net}")
        return o_net_boxes

    def __p_net_detect(self, image):
        bboxes = []

        scale = 1
        w, h = image.size
        min_side_len = min(w, h)
        while min_side_len >= 12:
            img_data = self.__image_transform(image)
            out_cls, out_offset = self.p_net(img_data.to(DEVICE).unsqueeze(0))
            out_cls, out_offset = out_cls[0][0].cpu().detach(), out_offset[0].cpu().detach()
            idxs = torch.nonzero(torch.gt(out_cls, 0.6))  # out_cls > 0.6  idxy[idx, idy]
            boxes = self.__box(idxs, out_offset, out_cls, scale)
            bboxes.extend(boxes)
            scale *= 0.709
            _w = int(w * scale)
            _h = int(h * scale)
            # 图像金字塔
            image = image.resize((_w, _h))
            min_side_len = min(_w, _h)  # 这里是跳出循环的条件

        return tools.nms(np.array(bboxes), 0.3, False)

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        _x1 = ((start_index[:, 1]).to(torch.float32) * stride) / scale
        _y1 = ((start_index[:, 0]).to(torch.float32) * stride) / scale
        _x2 = ((start_index[:, 1]).to(torch.float32) * stride + side_len) / scale
        _y2 = ((start_index[:, 0]).to(torch.float32) * stride + side_len) / scale

        ow, oh = side_len / scale, side_len / scale
        _offset = offset[:, start_index[:, 0], start_index[:, 1]]
        x1 = _x1 + ow * _offset[0, :]
        y1 = _y1 + oh * _offset[1, :]
        x2 = _x2 + ow * _offset[2, :]
        y2 = _y2 + oh * _offset[3, :]
        cls_ = cls[start_index[:, 0], start_index[:, 1]]
        bboxes = torch.stack([x1, y1, x2, y2, cls_], dim=1)
        return np.array(bboxes)

    def __ro_net_detect(self, image, net_boxes, face_size):
        print(face_size)
        _img_dataset = []
        _net_boxes = tools.convert_to_square(net_boxes)
        for _box in _net_boxes:  # _pnet_boxes [N, 5]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((face_size, face_size))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)  # [N, 3, 24, 24] (array([...], array([...]))
        img_dataset = torch.stack(_img_dataset)  # 有元组转换成张量
        img_dataset = img_dataset.to(DEVICE)
        if face_size == 24:
            out_cls, out_offset = self.r_net(img_dataset)  # [N, 1] [N, 4]
        elif face_size == 48:
            out_cls, out_offset = self.o_net(img_dataset)
        else:
            raise Exception("face_size not in [24, 48]!")
        out_cls = out_cls.cpu().detach().numpy()  # [N. 1]
        out_offset = out_offset.cpu().detach().numpy()  # [N, 4]

        idxs, _ = np.where(out_cls > 0.3)  # [N,]
        _boxes = _net_boxes[idxs]  # [N, 5]
        _x1 = _boxes[:, 0]
        _y1 = _boxes[:, 1]
        _x2 = _boxes[:, 2]
        _y2 = _boxes[:, 3]

        ow, oh = _x2 - _x1, _y2 - _y1

        x1 = _x1 + ow * out_offset[idxs][:, 0]  # [N, 1]
        y1 = _y1 + oh * out_offset[idxs][:, 1]  # [N, 1]
        x2 = _x2 + ow * out_offset[idxs][:, 2]  # [N, 1]
        y2 = _y2 + oh * out_offset[idxs][:, 3]  # [N, 1]
        cls = out_cls[idxs][:, 0]  # [N, 1]

        boxes = np.stack([x1, y1, x2, y2, cls], axis=1)
        if face_size == 24:
            return tools.nms(np.array(boxes), 0.3, False)
        else:
            return tools.nms(np.array(boxes), 0.3, True)


def show_single_image(image_file):
    with Image.open(image_file) as im:
        detector = Detector(
                 p_net_param=r"./save_params/p_net.pth",
                 r_net_param=r"./save_params/r_net.pth",
                 o_net_param="./save_params/o_net.pth")
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

