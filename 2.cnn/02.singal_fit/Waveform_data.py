import torch

# 指定N个原始数据作为一个周期的信号基础
seed_data = torch.randint(100, 200, (12,), dtype=torch.float32)

# 以一定的随机数扰动这个基础周期数据，使得数据具有一定的噪声
# 然后将生成的具有一定噪声的数据组合起来，形成周期数据

train_data = []

for i in range(50):
    gen_data = seed_data + torch.randint(-2, 2, (len(seed_data), ))
    train_data.append(gen_data)

#展开成一维数据
train_data=torch.stack(train_data).reshape(-1)

# 生成测试数据
test_data = []
for i in range(20):
    gen_data = seed_data+torch.randint(-2,2,(len(seed_data),))
    test_data.append(gen_data)
#展开成一维数据
test_data=torch.stack(test_data).reshape(-1)

print(train_data)
print(test_data)
print(train_data.shape)
print(test_data.shape)
torch.save(train_data,"./train.data")
torch.save(test_data,"./test.data")
