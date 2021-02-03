import re
import xml.etree.cElementTree as ET
import os
from cfg import CLASSES_NAME
import sys
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
from utils.tools import convert_to_416x416


def parse_xml(doc_save, xml_path=None, img_path=None):
    cls_dict = {}
    for i, cls in enumerate(CLASSES_NAME):
        cls_dict[cls] = i
    with open(xml_path, "r", encoding="utf-8") as f:
        xml = f.read()

    pattern1 = r"<path>(.*?)</path>"
    cls_name = r"<item>.*?<name>(.*?)</name>"
    xmin = r"<bndbox>.*?<xmin>(.*?)</xmin>"
    ymin = r"<bndbox>.*?<ymin>(.*?)</ymin>"
    xmax = r"<bndbox>.*?<xmax>(.*?)</xmax>"
    ymax = r"<bndbox>.*?<ymax>(.*?)</ymax>"
    res_filename = re.findall(re.compile(pattern1, re.S), xml)[0].split("\\")[-1]

    res_cls_name = re.findall(re.compile(cls_name, re.S), xml)
    res_xmin = re.findall(re.compile(xmin, re.S), xml)
    res_ymin = re.findall(re.compile(ymin, re.S), xml)
    res_xmax = re.findall(re.compile(xmax, re.S), xml)
    res_ymax = re.findall(re.compile(ymax, re.S), xml)
    data = np.stack([res_cls_name, res_xmin, res_ymin, res_xmax, res_ymax], axis=1)
    print(data)
    img_full_path = fr"G:\3class_2_group/img/{res_filename}"
    img = Image.open(img_full_path)
    index = res_filename.split(".")[0]

    _, _, s, _ = convert_to_416x416(img, 416, img_path, index)

    doc_save.write(f"{res_filename} ")
    for cls in data:
        cls_name, x1, y1, x2, y2 = cls[0], float(cls[1]), float(cls[2]), float(cls[3]), float(cls[4])
        x1, y1, x2, y2 = x1 * s, y1 * s, x2 * s, y2 * s
        cx, cy, w, h = x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, x2 - x1, y2 - y1
        # if cls_name == "猎豹":
        #     cls_name = "豹子"
        doc_save.write(f"{cls_dict[cls_name]} {float(cx)} {float(cy)} {float(w)} {float(h)} ")
        doc_save.flush()
    doc_save.write("\n")
    doc_save.flush()


if __name__ == '__main__':
    save_path = r"../data/lable_class_3.txt"
    img_path = r"G:\3class_2_group\process_img"
    xml_path = r"G:\3class_2_group\anno"
    r"G:\3class_2_group\process_img"
    file = open(save_path, "w").close()  # 清空
    doc_save = open(save_path, "w+")
    xml_total = os.listdir(xml_path)

    length = len(xml_total)
    for i in tqdm(range(length)):
        xml_single_path = f"{xml_path}/{xml_total[i]}"
        parse_xml(doc_save, xml_single_path, img_path)
    doc_save.close()
