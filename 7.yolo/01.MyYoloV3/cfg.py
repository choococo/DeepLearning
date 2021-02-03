import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_HEIGHT = 416
IMG_WIDTH = 416
# CLASS_NUM = 20  # 使用voc数据集

"anchor box 是对coco数据集聚类获得"
# ANCHORS_GROUP = {
#     13: [[116, 90], [156, 198], [373, 326]],
#     26: [[30, 61], [62, 45], [59, 119]],
#     52: [[10, 13], [16, 30], [33, 23]]
# }

ANCHORS_GROUP = {
    13: [[360, 360], [360, 180], [180, 360]],
    26: [[180, 180], [180, 90], [90, 180]],
    52: [[90, 90], [90, 45], [45, 90]]
}
# ANCHORS_GROUP = {
#     52: [[14, 17], [25, 40], [40, 85]],
#     26: [[62, 44], [69, 152], [109, 92]],
#     13: [[135, 223], [238, 140], [296, 277]]
# }
ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}

# CLASSES_NAME = [
#     "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# ]
# CLASSES_NAME = [
#     "人", "狮子", "老虎", "豹子", "狗"
# ]

CLASSES_NAME = [
    "人", "老虎", "熊猫"
]

CLASS_NUM = len(CLASSES_NAME)

if __name__ == '__main__':
    for feature_size, anchors in ANCHORS_GROUP.items():
        print(feature_size)
        print(anchors)

    for feature_size, anchor_area in ANCHORS_GROUP_AREA.items():
        print(feature_size)
        print(anchor_area)
