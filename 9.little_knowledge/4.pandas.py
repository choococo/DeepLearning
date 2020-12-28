import pandas as pd

"""
pandas 使用：
    （1）读取文件操作
"""
read_path = r"F:\workspace\MTCNN\01.mtcnn\label\list_bbox_celeba.txt"  # ctrl+shift+c 复制绝对路径
# header=1:跳过第0行，从第一行开始读；delim_whitesapce=True: 未知，先加上；index_col=0：本来有默认的行索引，现在将第一列作为行索引
a = pd.read_table(read_path, header=1, delim_whitespace=True, index_col=0)
print(a)
"""
             x_1  y_1  width  height
image_id                            
000001.jpg    95   71    226     313
000002.jpg    72   94    221     306
000003.jpg   216   59     91     126
000004.jpg   622  257    564     781
000005.jpg   236  109    120     166
...          ...  ...    ...     ...
202595.jpg  1381   91    221     306
202596.jpg   137  129    114     158
202597.jpg    53   76     91     126
202598.jpg   195   28     91     126
202599.jpg   101  101    179     248

[202599 rows x 4 columns]
"""
# 拿到第一行的数据,iloc是强制索引，与loc不同
print(a.iloc[0])
print(a.loc['000001.jpg'])
"""
[202599 rows x 4 columns]
x_1        95
y_1        71
width     226
height    313
Name: 000001.jpg, dtype: int64
"""

# 拿到索引
print(a.index)
"""
Index(['000001.jpg', '000002.jpg', '000003.jpg', '000004.jpg', '000005.jpg',
       '000006.jpg', '000007.jpg', '000008.jpg', '000009.jpg', '000010.jpg',
       ...
       '202590.jpg', '202591.jpg', '202592.jpg', '202593.jpg', '202594.jpg',
       '202595.jpg', '202596.jpg', '202597.jpg', '202598.jpg', '202599.jpg'],
      dtype='object', name='image_id', length=202599)
"""
