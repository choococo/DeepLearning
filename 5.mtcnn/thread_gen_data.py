import threading
import time
from gen_sample import gen_data

'多线程生成数据'


def gen_train_data(index):
    print("生成训练数据")
    choices = {0: "train", 1: "val", 2: "test"}
    label_position_path = fr"new_label/{choices[0]}_position.txt"
    label_landmark_path = fr"new_label/{choices[0]}_landmark.txt"
    img_path = r"F:\2.Dataset\img_celeba"

    # label_position_path = f"label/list_bbox_celeba.txt"
    # label_landmark_path = f"label/list_landmarks_celeba.txt"
    # img_path = r"image"

    save_path = fr"F:\2.Dataset\mtcnn_dataset\testing\saving\{choices[0]}"
    gen_data(img_path, label_position_path, label_landmark_path, save_path)


def gen_val_data(index):
    print("生成验证数据")
    choices = {0: "train", 1: "val", 2: "test"}
    label_position_path = fr"new_label/{index}_position.txt"
    label_landmark_path = fr"new_label/{index}_landmark.txt"
    img_path = r"F:\2.Dataset\img_celeba"

    # label_position_path = f"label/list_bbox_celeba.txt"
    # label_landmark_path = f"label/list_landmarks_celeba.txt"
    # img_path = r"image"

    save_path = fr"F:\2.Dataset\mtcnn_dataset\testing\saving\{index}"
    gen_data(img_path, label_position_path, label_landmark_path, save_path)


def gen_test_data(index):
    print("生成测试数据")
    choices = {0: "train", 1: "val", 2: "test"}
    label_position_path = fr"new_label/{index}_position.txt"
    label_landmark_path = fr"new_label/{index}_landmark.txt"
    img_path = r"F:\2.Dataset\img_celeba"

    # label_position_path = f"label/list_bbox_celeba.txt"
    # label_landmark_path = f"label/list_landmarks_celeba.txt"
    # img_path = r"image"

    save_path = fr"F:\2.Dataset\mtcnn_dataset\testing\saving\{index}"
    gen_data(img_path, label_position_path, label_landmark_path, save_path)


if __name__ == '__main__':
    choices = {0: "train", 1: "val", 2: "test"}

    thread1 = threading.Thread(target=gen_train_data, args=(choices[0],), name="线程1")
    thread2 = threading.Thread(target=gen_val_data, args=(choices[1],), name="线程2")
    thread3 = threading.Thread(target=gen_test_data, args=(choices[2],), name="线程3")
    thread1.start()
    thread2.start()
    thread3.start()
