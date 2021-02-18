import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
from mtcnn import tools, net
import time
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Detector:

    def __init__(self, p_net_param=r"mtcnn\params_p/p_net_52_s.pth",
                 # 61，
                 r_net_param="mtcnn/params_r/r_net_61.pth",
                 o_net_param=r"mtcnn/params_o/o_net_104.pth", isCuda=False):
        self.isCuda = isCuda
        self.p_net = net.PNet().to(DEVICE)
        self.r_net = net.RNet().to(DEVICE)
        self.o_net = net.ONet().to(DEVICE)

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
            # out_cls[N, 1, H, W]  out_offset[N. 4. H. W] 去掉梯度
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
        # 将P网络的偏移量反算到原图，得到框在原图上的偏移量
        _x1 = ((start_index[:, 1]).to(torch.float32) * stride) / scale
        _y1 = ((start_index[:, 0]).to(torch.float32) * stride) / scale
        _x2 = ((start_index[:, 1]).to(torch.float32) * stride + side_len) / scale
        _y2 = ((start_index[:, 0]).to(torch.float32) * stride + side_len) / scale

        # ow, oh = _x2 - _x1, _y2 - _y1
        ow, oh = side_len / scale, side_len / scale
        # 偏移框这个时候相当于建议框
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
        # 将所有的net的box转成正方形
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
            # print(out_offset.shape)
            # out_offset = out_offset[:5]
        elif face_size == 48:
            out_cls, out_offset = self.o_net(img_dataset)
        else:
            raise Exception("face_size not in [24, 48]!")
        out_cls = out_cls.cpu().detach().numpy()  # [N. 1]
        out_offset = out_offset.cpu().detach().numpy()  # [N, 4]

        # 选取符合条件的索引
        if face_size == 24:
            idxs, _ = np.where(out_cls > 0.7)  # [N,]
        elif face_size == 48:
            idxs, _ = np.where(out_cls > 0.9999)  # [N,]
        else:
            raise Exception("face size must be in [24, 48]")
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
        if face_size == 24:
            boxes_24 = np.stack([x1, y1, x2, y2, cls], axis=1)
            return tools.nms(np.array(boxes_24), 0.3, False)
        elif face_size == 48:
            px1 = _x1 + ow * out_offset[idxs][:, 4]  # [N, 1]
            py1 = _y1 + oh * out_offset[idxs][:, 5]  # [N, 1]
            px2 = _x1 + ow * out_offset[idxs][:, 6]  # [N, 1]
            py2 = _y1 + oh * out_offset[idxs][:, 7]  # [N, 1]
            px3 = _x1 + ow * out_offset[idxs][:, 8]  # [N, 1]
            py3 = _y1 + oh * out_offset[idxs][:, 9]  # [N, 1]
            px4 = _x1 + ow * out_offset[idxs][:, 10]  # [N, 1]
            py4 = _y1 + oh * out_offset[idxs][:, 11]  # [N, 1]
            px5 = _x1 + ow * out_offset[idxs][:, 12]  # [N, 1]
            py5 = _y1 + oh * out_offset[idxs][:, 13]  # [N, 1]
            boxes_48 = np.stack([x1, y1, x2, y2, cls, px1, py1, px2, py2, px3, py3, px4, py4, px5, py5], axis=1)
            # boxes_48 = np.stack([x1, y1, x2, y2, cls], axis=1)
            return tools.nms(np.array(boxes_48), 0.3, True)
        else:
            raise Exception("face size must be in [24, 48")


def show_single_image(image_file):
    with Image.open(image_file) as im:
        detector = Detector(p_net_param=r".\params_p/p_net_52_s.pth",
                            # 61，
                            r_net_param="./params_r/r_net_61.pth",
                            o_net_param=r"./params_o/o_net_74.pth")
        boxes = detector.detect(im)
        print(boxes.shape)
        imDraw = ImageDraw.Draw(im)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            px1 = int(box[5])
            py1 = int(box[6])
            px2 = int(box[7])
            py2 = int(box[8])
            px3 = int(box[9])
            py3 = int(box[10])
            px4 = int(box[11])
            py4 = int(box[12])
            px5 = int(box[13])
            py5 = int(box[14])
            print(box[4])
            imDraw.rectangle((x1, y1, x2, y2), outline='red', width=3)
            imDraw.ellipse((px1, py1, px1 + 3, py1 + 3), fill='red')
            imDraw.ellipse((px2, py2, px2 + 3, py2 + 3), fill='red')
            imDraw.ellipse((px3, py3, px3 + 3, py3 + 3), fill='red')
            imDraw.ellipse((px4, py4, px4 + 3, py4 + 3), fill='red')
            imDraw.ellipse((px5, py5, px5 + 3, py5 + 3), fill='red')
        y = time.time()
        # print(y - x)
        im.show()
        im.save("../save_image2.jpg")


if __name__ == '__main__':
    x = time.time()
    with torch.no_grad() as grad:
        image_file = r"images\2.jpg"
        # image_file = r"F:\2.Dataset\img_celeba/000001.jpg"
        image_file = r"images2\10.jpg"
        show_single_image(image_file)
        exit()
        image_path = r"images"
        for i in os.listdir(image_path):
            detector = Detector()
            with Image.open(os.path.join(image_path, i)) as im:

                # w, h = im.size
                #
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
