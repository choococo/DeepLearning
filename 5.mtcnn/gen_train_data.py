from gen_sample import gen_data


if __name__ == '__main__':
    choices = {0: "train", 1: "val", 2: "test"}
    # label_position_path = fr"new_label/{choices[0]}_position.txt"
    # label_landmark_path = fr"new_label/{choices[0]}_landmark.txt"
    # img_path = r"image"

    label_position_path = f"label/list_bbox_celeba.txt"
    label_landmark_path = f"label/list_landmarks_celeba.txt"
    img_path = r"image"

    save_path = fr"F:\2.Dataset\mtcnn_dataset\testing\saving\{choices[2]}"
    gen_data(img_path, label_position_path, label_landmark_path, save_path)