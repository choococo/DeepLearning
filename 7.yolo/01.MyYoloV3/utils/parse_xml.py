import re
import xml.etree.cElementTree as ET
import os
from cfg import CLASSES_NAME
import sys
from tqdm import tqdm
import numpy as np
from utils.tools import convert_to_416x416
from PIL import Image


def parse_xml(doc_save, xml_path=None, img_path=None, process_path=None):
    cls_dict = {}
    for i, cls in enumerate(CLASSES_NAME):
        cls_dict[cls] = i
    with open(xml_path, "r") as f:
        xml = f.read()

    pattern1 = r"<filename>(.*?)</filename>"
    cls_name = r"<object>.*?<name>(.*?)</name>"
    xmin = r"<object>.*?<xmin>(.*?)</xmin>"
    ymin = r"<object>.*?<ymin>(.*?)</ymin>"
    xmax = r"<object>.*?<xmax>(.*?)</xmax>"
    ymax = r"<object>.*?<ymax>(.*?)</ymax>"
    res_filename = re.findall(re.compile(pattern1, re.S), xml)
    res_cls_name = re.findall(re.compile(cls_name, re.S), xml)
    res_xmin = re.findall(re.compile(xmin, re.S), xml)
    res_ymin = re.findall(re.compile(ymin, re.S), xml)
    res_xmax = re.findall(re.compile(xmax, re.S), xml)
    res_ymax = re.findall(re.compile(ymax, re.S), xml)
    data = np.stack([res_cls_name, res_xmin, res_ymin, res_xmax, res_ymax], axis=1)
    img_full_path = fr"{img_path}/{res_filename[0]}"
    img = Image.open(img_full_path)
    index = res_filename[0].split(".")[0]

    _, _, s, _ = convert_to_416x416(img, 416, process_path, index)

    doc_save.write(f"{res_filename[0]} ")
    for cls in data:
        cls_name, x1, y1, x2, y2 = cls[0], float(cls[1]), float(cls[2]), float(cls[3]), float(cls[4])
        x1, y1, x2, y2 = x1 * s, y1 * s, x2 * s, y2 * s
        cx, cy, w, h = x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, x2 - x1, y2 - y1
        doc_save.write(f"{cls_dict[cls_name]} {float(cx)} {float(cy)} {float(w)} {float(h)} ")
        doc_save.flush()
    doc_save.write("\n")
    doc_save.flush()


if __name__ == '__main__':
    # save_path = r"../data/lable3.txt"
    # img_path = r"F:\BaiduNetdiskDownload\VOCtrainval_11-May-2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages"
    # xml_path = r"F:\BaiduNetdiskDownload\VOCtrainval_11-May-2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations"
    # process_img = r"F:\Dataset03\VOC\process_img"
    # file = open(save_path, "w").close()  # 清空
    # doc_save = open(save_path, "w+")
    # xml_total = os.listdir(xml_path)
    #
    # length = len(xml_total)
    # for i in tqdm(range(length)):
    #     xml_single_path = f"{xml_path}/{xml_total[i]}"
    #     parse_xml(doc_save, xml_single_path, img_path, process_img)
    # doc_save.close()
    import torch
    print(torch.cuda.get_device_properties(0))
    torch.cuda.is_available()
