a = "000001.jpg    95  71 226 313"
print(a)

print(a.strip().split(" "))
print(a.split(" "))
#
b = a.split(" ")
# 返回一个迭代器，不为空的值
print(list(filter(bool, b)))
print(a.split())

x = None
print(bool())
print(bool(x))
print(bool("fdfd"))
