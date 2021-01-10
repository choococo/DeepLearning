a = [1, 2, 3]

a.append(4)
print(a)

b = [5, 6]

# 展平成当前维的的数据
a.extend(b)
print(a)

a.append(b)
print(a)
