import os
import numpy as np
import pandas as pd

"""
将一个文件的内容按行追加到每行后面形成一一对应的,进行拼接
文件读写操作的时候，一定要注意读写分离
"""
origin_path = r"file/sample1.txt"
append_path = r"file/sample2.txt"
save_path = r"file/sample3.txt"

with open(origin_path, "r") as f1, open(append_path, "r") as f2:  # 同时打开两个文件
    str1 = [i.strip().split() for i in f1.readlines()]
    str2 = [i.strip().split()[1:] for i in f2.readlines()]
    print(str1)
    print(str2)
    str3 = np.concatenate([str1, str2], axis=1)
    print(str3)

with open(save_path, "w") as f:
    for i in range(len(str3)):
        res = " ".join(str3[i])
        f.writelines(res)
        f.writelines("\n")
        f.flush()


with open(origin_path, "r") as f1, open(append_path, "r") as f2:  # 同时打开两个文件
    str1 = [i.strip().split() for i in f1.readlines()]
    str2 = [i.strip().split()[1:] for i in f2.readlines()]
    print(str1)
    print(str2)
    str3 = np.concatenate([str1, str2], axis=1)
    print(str3)

"""
[['image_name', 'x1', 'y1', 'x2', 'y2'], ['0.jpg', '123', '435', '74', '23'], 
['1.jpg', '123', '435', '74', '23'], ['2.jpg', '123', '435', '74', '23'], ['3.jpg', '123', '435', '74', '23'], 
['4.jpg', '123', '435', '74', '23'], ['5.jpg', '123', '435', '74', '23'], ['6.jpg', '123', '435', '74', '23']]

[['lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y', 
'rightmouth_x', 'rightmouth_y'], ['123', '124', '345', '64', '34', '74', '23', '89', '63', '23'], 
['123', '124', '345', '64', '34', '74', '23', '89', '63', '23'], 
['123', '124', '345', '64', '34', '74', '23', '89', '63', '23'], 
['123', '124', '345', '64', '34', '74', '23', '89', '63', '23'],
 ['123', '124', '345', '64', '34', '74', '23', '89', '63', '23'], 
 ['123', '124', '345', '64', '34', '74', '23', '89', '63', '23'], 
 ['123', '124', '345', '64', '34', '74', '23', '89', '63', '23']]
 
 
[['image_name' 'x1' 'y1' 'x2' 'y2' 'lefteye_x' 'lefteye_y' 'righteye_x'
  'righteye_y' 'nose_x' 'nose_y' 'leftmouth_x' 'leftmouth_y'
  'rightmouth_x' 'rightmouth_y']
 ['0.jpg' '123' '435' '74' '23' '123' '124' '345' '64' '34' '74' '23'
  '89' '63' '23']
 ['1.jpg' '123' '435' '74' '23' '123' '124' '345' '64' '34' '74' '23'
  '89' '63' '23']
 ['2.jpg' '123' '435' '74' '23' '123' '124' '345' '64' '34' '74' '23'
  '89' '63' '23']
 ['3.jpg' '123' '435' '74' '23' '123' '124' '345' '64' '34' '74' '23'
  '89' '63' '23']
 ['4.jpg' '123' '435' '74' '23' '123' '124' '345' '64' '34' '74' '23'
  '89' '63' '23']
 ['5.jpg' '123' '435' '74' '23' '123' '124' '345' '64' '34' '74' '23'
  '89' '63' '23']
 ['6.jpg' '123' '435' '74' '23' '123' '124' '345' '64' '34' '74' '23'
  '89' '63' '23']]
"""
