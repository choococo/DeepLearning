import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
from utils import tools
import net
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Detector:

    def __init__(self, p_net_param="./params_p/p_net.pth", r_net_param="./params_r/r_net.pth",
                 o_net_param="./params_o/o_net.pth", isCuda=False):
        self.isCuda = isCuda  # 判断是否有cuda
        self.p_net = net.PNet().to(DEVICE)
        self.r_net = net.RNet().to(DEVICE)
        self.o_net = net.ONet().to(DEVICE)

        self.p_net.load_state_dict(torch.load(p_net_param))  # 加载参数
        self.r_net.load_state_dict(torch.load(r_net_param))
        self.o_net.load_state_dict(torch.load(o_net_param))
        self.p_net.eval()  # 进入测试模式
        self.r_net.eval()
        self.o_net.eval()
        self.__image_transform = transforms.Compose([  # 图片数据处理
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对于图像这个是有经验值的
        ])

    def detect(self, image):  # 侦测
        start_time = time.time()
        p_net_boxes = self.__p_net_detect(image)  # 进入p网络
        if p_net_boxes.shape[0] == 0:  # 如果没有框，直接返回空的数组
            return np.array([])
        end_time = time.time()
        t_p_net = end_time - start_time

        start_time = time.time()
        r_net_boxes = self.__r_net_detect(image, p_net_boxes)  # 进入r网络
        if r_net_boxes.shape[0] == 0:  # 如果没有框，直接返回空的数组
            return np.array([])
        end_time = time.time()
        t_r_net = end_time - start_time

        start_time = time.time()
        o_net_boxes = self.__o_net_detect(image, r_net_boxes)  # 进入o网络
        if o_net_boxes.shape[0] == 0:  # 如果没有框，直接返回空的数组
            return np.array([])
        end_time = time.time()
        t_o_net = end_time - start_time
        total_time = t_p_net + t_r_net + t_o_net  # 计算总时间
        print(f"[time]  total:{total_time} p_net:{t_p_net} r_net:{t_r_net} o_net:{t_o_net}")

    def __p_net_detect(self, image):  # p网络侦测
        boxes = []  # 最终所有的框

        scale = 1  # 缩放比例，最开始的时候为1
        w, h = image.size  # 获得图片的宽和高
        min_side_len = min(w, h)  # 获取宽和高中的最小值，作为图像金字塔的最小边长(停止条件)
        while min_side_len >= 12:  # 开始进行图像金字塔
            img_data = self.__image_transform(image)  # 对数据进行处理

            out_cls, out_offset = self.p_net(img_data.to(DEVICE).unsqueeze(0))  # 网络输出
            # out_cls[N, 1, H, W]  out_offset[N. 4. H. W] 去掉梯度, 取到相对应的值
            out_cls, out_offset = out_cls[0][0].cpu().detach(), out_offset[0].cpu().detach()
            idxs = torch.nonzero(torch.gt(out_cls, 0.6))  # out_cls > 0.6  idxy[idx, idy]
            boxes = self.__box(idxs, out_offset, out_cls, scale)  # 反算坐标到原图
            scale *= 0.709  # 新的缩放比例
            _w = int(w * scale)  # 获得缩放比例后的w
            _h = int(h * scale)  # 获得缩放比例后的h

            # 图像金字塔
            image = image.resize((_w, _h))
            min_side_len = min(_w, _h)  # 这里是跳出循环的条件

        return tools.nms(np.array(boxes), 0.3, False)  # 计算nms

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        # 将P网络的偏移量反算到原图，得到框在原图上的偏移量
        _x1 = ((start_index[:, 1]).to(torch.float32) * stride) / scale  # 根据索引反算到原图中
        _y1 = ((start_index[:, 0]).to(torch.float32) * stride) / scale
        _x2 = ((start_index[:, 1]).to(torch.float32) * stride + side_len) / scale
        _y2 = ((start_index[:, 0]).to(torch.float32) * stride + side_len) / scale

        ow, oh = _x2 - _x1, _y2 - _y1
        # 偏移框这个时候相当于建议框
        _offset = offset[:, start_index[:, 0], start_index[:, 1]]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]
        cls_ = cls[start_index[:, 0], start_index[:, 1]]  # 获取置信度
        bboxes = torch.stack([x1, y1, x2, y2, cls_], dim=1)  # 组合
        return bboxes

    def __r_net_detect(self, image, pnet_boxes):
        _img_dataset = []
        # 将所有的pnet的box转成正方形
        _pnet_boxes = tools.convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:  # _pnet_boxes [N, 5]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            img = image.crop((24, 24))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)  # [N, 3, 24, 24] (array([...], array([...]))
        img_dataset = torch.stack(_img_dataset)  # 有元组转换成张量
        img_dataset = img_dataset.to(DEVICE)
        out_cls, out_offset = self.r_net(img_dataset)  # [N, 1] [N, 4]
        out_cls = out_cls.cpu().detach().numpy()  # [N. 1]
        out_offset = out_offset.cpu().detach().numpy()  # [N, 4]

        boxes = []
        # 选取符合条件的索引
        idxs, _ = np.where(out_cls > 0.6)  # [N,]
        _boxes = _pnet_boxes[idxs]  # [N, 5]
        _x1 = int(_boxes[:, 0])
        _y1 = int(_boxes[:, 1])
        _x2 = int(_boxes[:, 2])
        _y2 = int(_boxes[:, 3])

        ow, oh = _x2 - _x1, _y2 - _y1

        x1 = _x1 + ow * out_offset[idxs][:, 0]  # [N, 1]
        y1 = _y1 + oh * out_offset[idxs][:, 1]  # [N, 1]
        x2 = _x2 + ow * out_offset[idxs][:, 2]  # [N, 1]
        y2 = _y2 + oh * out_offset[idxs][:, 3]  # [N, 1]
        cls = out_cls[idxs][:, 0]  # [N, 1]

        boxes = np.stack([x1, y1, x2, y2, cls], axis=1)
        return tools.nms(np.array(boxes), 0.3, False)

    def __o_net_detect(self, image, rnet_boxes):
        _img_dataset = []
        _rnet_boxes = tools.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:  # [N, 5]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)  # [H,W,C]->[C,H,W]
            _img_dataset.append(img_data)  # [tensor(...), tensor(...)]
        img_dataset = torch.stack(_img_dataset)
        img_dataset = img_dataset.to(DEVICE)
        out_cls, out_offset = self.o_net(img_dataset)  # [N, 1] [N. 4]

        # 放到cpu上
        out_cls = out_cls.cpu().detach().numpy()
        out_offset = out_offset.cpu().detach().numpy()

        boxes = []
        # 根据置信度筛选格子
        idxs, _ = np.where(out_cls > 0.90)
        _boxes = _rnet_boxes[idxs]  # 在正方形的框中筛选 [N]
        _x1 = int(_boxes[:, 0])
        _y1 = int(_boxes[:, 1])
        _x2 = int(_boxes[:, 2])
        _y2 = int(_boxes[:, 3])
        cls = out_cls[idxs][:, 0]
        # cls = out_cls[idxs].reshape(-1) # 和上面的用法相同

        boxes = np.stack([_x1, _y1, _x2, _y2, cls], axis=1)
        return tools.nms(np.array(boxes), 0.3, isMin=True)

    def __ro_net_detect(self, image, net_boxes, face_size=24):
        _img_dataset = []
        # 将所有的net的box转成正方形
        _net_boxes = tools.convert_to_square(net_boxes)
        for _box in _net_boxes:  # _pnet_boxes [N, 5]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            img = image.crop((24, 24))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)  # [N, 3, 24, 24] (array([...], array([...]))
        img_dataset = torch.stack(_img_dataset)  # 有元组转换成张量
        img_dataset = img_dataset.to(DEVICE)
        out_cls, out_offset = self.r_net(img_dataset)  # [N, 1] [N, 4]
        out_cls = out_cls.cpu().detach().numpy()  # [N. 1]
        out_offset = out_offset.cpu().detach().numpy()  # [N, 4]

        boxes = []
        # 选取符合条件的索引
        idxs, _ = np.where(out_cls > 0.6)  # [N,]
        _boxes = _net_boxes[idxs]  # [N, 5]
        _x1 = int(_boxes[:, 0])
        _y1 = int(_boxes[:, 1])
        _x2 = int(_boxes[:, 2])
        _y2 = int(_boxes[:, 3])

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
            # face_size = 48
            return tools.nms(np.array(boxes), 0.3, True)

