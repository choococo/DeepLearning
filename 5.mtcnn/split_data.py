choices = {0: "train", 1: "val", 2: "test"}

train_position_path = fr"new_label/{choices[0]}_position.txt"
train_landmark_path = fr"new_label/{choices[0]}_landmark.txt"

val_position_path = fr"new_label/{choices[1]}_position.txt"
val_landmark_path = fr"new_label/{choices[1]}_landmark.txt"

test_position_path = fr"new_label/{choices[2]}_position.txt"
test_landmark_path = fr"new_label/{choices[2]}_landmark.txt"

bbox_list = []
landmark_list = []
with open("label/list_landmarks_celeba.txt", "r") as f:
    for i in f.readlines()[2:]:
        line = i.strip().split()
        bbox_list.append(line)
print(bbox_list)
print(len(bbox_list))

train_num = int(len(bbox_list) * 0.7)
val_num = int(len(bbox_list) * 0.15) + 1
test_num = int(len(bbox_list) - train_num - val_num)
print(train_num, val_num, test_num)

# with open(train_landmark_path, "w") as f:
#     for i in bbox_list[:train_num]:
#         img_name, fx1, fy1, fx2, fy2, fx3, fy3, fx4, fy4, fx5, fy5 = i
#         f.writelines(f"{img_name} {fx1} {fy1} {fx2} {fy2} {fx3} {fy3} {fx4} {fy4} {fx5} {fy5}")
#         f.write("\n")
#         f.flush()
with open(val_landmark_path, "w") as f:
    for i in bbox_list[train_num:train_num+val_num]:
        img_name, fx1, fy1, fx2, fy2, fx3, fy3, fx4, fy4, fx5, fy5 = i
        f.writelines(f"{img_name} {fx1} {fy1} {fx2} {fy2} {fx3} {fy3} {fx4} {fy4} {fx5} {fy5}")
        f.write("\n")
        f.flush()
with open(test_landmark_path, "w") as f:
    for i in bbox_list[train_num+val_num:]:
        img_name, fx1, fy1, fx2, fy2, fx3, fy3, fx4, fy4, fx5, fy5 = i
        f.writelines(f"{img_name} {fx1} {fy1} {fx2} {fy2} {fx3} {fy3} {fx4} {fy4} {fx5} {fy5}")
        f.write("\n")
        f.flush()
# with open(val_landmark_path, "w") as f:
#     for i in bbox_list[train_num:train_num+val_num]:
#         img_name, x1, y1, w, h = i
#         f.writelines(f"{img_name} {x1} {y1}  {w} {h}")
#         f.write("\n")
#         f.flush()
# with open(test_landmark_path, "w") as f:
#     for i in bbox_list[train_num+val_num:]:
#         img_name, x1, y1, w, h = i
#         f.writelines(f"{img_name} {x1} {y1}  {w} {h}")
#         f.write("\n")
#         f.flush()
exit()
