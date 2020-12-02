import os
import shutil
import random

"对猫狗二分类数据集进行划分，划分为train,val,test"
"""
划分
"""

path = r"F:\2.Dataset\cat_dog\dog"
save_path = r"F:\2.Dataset\cat_dog\test"

image_filename_list = []
for i, image in enumerate(os.listdir(path)):
    # print(i)
    # print(image)
    image_filename_list.append(image)
# print(image_filename_list)

res = random.sample(image_filename_list, 899)
print(res)

for img_name in res:
    shutil.move(os.path.join(path, img_name), os.path.join(save_path, img_name))

print("end")
