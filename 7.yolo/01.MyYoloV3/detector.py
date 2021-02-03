from darkent53_2 import MainNet
# from darknet53 import MainNet
import torch
import torch.nn as nn
import cfg
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from utils import tools
import os
import matplotlib.pyplot as plt

img_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Detector(nn.Module):

    def __init__(self, save_path):
        super(Detector, self).__init__()
        self.net = MainNet(cfg.CLASS_NUM).cuda()
        self.net.load_state_dict(torch.load(save_path))
        self.net.eval()

    def forward(self, x, thresh, anchors):
        output_13, output_26, output_52 = self.net(x)
        idx_s_13, vec_s_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idx_s_13, vec_s_13, 32, anchors[13])
        idx_s_26, vec_s_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idx_s_26, vec_s_26, 16, anchors[26])
        idx_s_52, vec_s_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idx_s_52, vec_s_52, 8, anchors[52])
        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

    @staticmethod
    def _filter(output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        # output.shape=[N, 13, 13, 3, 10]  10:[c, cx, cy, w, h, cls:]
        mask_obj = torch.sigmoid(output[..., 0]) > thresh
        idx_s = mask_obj.nonzero(as_tuple=False)
        vec_s = output[mask_obj]
        return idx_s, vec_s

    @staticmethod
    def _parse(idxs, vecs, t, anchors):  # idxs[N, 4] vecs[N,15]
        # 主要用于反算
        if len(idxs) == 0:
            return torch.randn(0, 6).cuda()
        anchors = torch.tensor(anchors, dtype=torch.float32).cuda()
        # idxs.shape=[N, 13, 13, 3]  3:3种建议框
        a = idxs[:, 3]
        confidence = torch.sigmoid(vecs[:, 0])
        _classify = vecs[:, 5:]
        classify = torch.argmax(_classify, dim=1).float()
        # idxs[A, 4] 4:[N,H,W,3]  vecs[N, 15(c-cx-cy-w-h-cls)]
        cx = (idxs[:, 2].float() + torch.sigmoid(vecs[:, 1])) * t
        cy = (idxs[:, 1].float() + torch.sigmoid(vecs[:, 2])) * t
        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[a, 1] * torch.exp(vecs[:, 4])

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = x1 + w
        y2 = y1 + h

        return torch.stack([confidence, x1, y1, x2, y2, classify], dim=1)


if __name__ == '__main__':
    # checkpoints = r"F:\workspace\7.YOLO\01.MyYolov3\checkpoints\net_yolo_72.pt"
    checkpoints = r"checkpoints_better\yolo_3cls_123.pt"
    img_path = r"G:\3class_2_group\img"
    detector = Detector(save_path=checkpoints)
    font = ImageFont.truetype("msyh.ttc", 20)
    for i in os.listdir(img_path):

        # image = Image.open(f"{img_path}/{i}")
        image = Image.open(r"F:\workspace\7.YOLO\01.MyYolov3\data\images\1.jpg")

        # image = Image.open(r"1.jpg")
        draw = ImageDraw.Draw(image)
        _, _, s, pro_image = tools.convert_to_416x416(image, 416)
        img_tensor = img_transform(pro_image)[None, ...].cuda()
        out_value = detector(img_tensor, 0.2, cfg.ANCHORS_GROUP).cpu().detach()
        boxes = []

        for j in range(cfg.CLASS_NUM):
            classify_mask = (out_value[..., -1] == j)
            _boxes = out_value[classify_mask]
            if _boxes.shape[0] != 0:
                boxes.append(tools.nms(_boxes))
        for box in boxes:
            try:
                for i in box:
                    c, xx1, yy1, xx2, yy2 = i[0:5]
                    cls_ = i[5]
                    # print(cls_)
                    print(c, xx1, yy1, xx2, yy2)
                    draw.rectangle((xx1 / s, yy1 / s, xx2 / s, yy2 / s))
                    draw.rectangle((xx1 / s, yy1 / s - 18, xx2 / s, yy1 / s), fill="orange")

                    draw.text((xx1 / s, yy1 / s - 23), cfg.CLASSES_NAME[int(cls_)], fill='white', font=font)

            except:
                continue

        image.show()
        exit()
        plt.imshow(image)
        plt.pause(1)
