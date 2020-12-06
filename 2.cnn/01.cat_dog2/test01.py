str1 = 0.679444432258606
str2 = 0.679444432258606
str3 = 0.679444432258606
str4 = "params"
str5 = "params1"


print(f"{str4:10s}{str1:18f}{str2:18f}{str3:18f}")
# print(f"{str5:10s}{str1:0d}{str2:9d}{str3:9d}")


import pandas as pd

data = pd.read_table("summaryWriter.txt", sep=" ", header=None)
print(data)
ax = data.iloc[:, [0,1]]
print(ax)


