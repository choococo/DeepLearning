from torchvision import datasets, transforms
from PIL import Image
import os


def make_dir(save_path, train_path, test_path):
    """
    创建所需文件夹
    :return: null
    """
    try:
        for dir in [save_path, train_path, test_path]:
            if not os.path.exists(dir):
                os.mkdir(dir)
        for i in range(10):
            if not os.path.exists(os.path.join(train_path, str(i))):
                os.mkdir(os.path.join(train_path, str(i)))
            if not os.path.exists(os.path.join(test_path, str(i))):
                os.mkdir(os.path.join(test_path, str(i)))
    except Exception as e:
        print(e)


def generator(dataset, data_path, interval=10, isShow=True):
    """
    生成手MNIST数据，图片格式，保存
    :param dataset: 训练集或者测试集
    :param interval: 每隔多少张图片显示一次
    :param data_path: 训练集或者测试集保存路径
    :param isShow: 是否显示图片
    :return: 保存成功，显示数据条数
    """
    _dict = {j: 0 for j in range(10)}
    print(_dict)
    for i, (image, label) in enumerate(dataset):
        _dict[label] += 1
        img = transforms.ToPILImage()(image)
        img.save(f"{data_path}/{label}/{_dict[label]}.png")
        if isShow and i % interval == 0:
            img.show()
    print(sum(_dict.values()))


if __name__ == '__main__':
    train_dataset = datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST("./data", train=False, transform=transforms.ToTensor(), download=False)

    print(train_dataset.data.shape)
    print(train_dataset.targets.shape)

    save_path = r"F:\2.Dataset\MNIST_DATASET"
    train_path = os.path.join(save_path, "train")
    test_path = os.path.join(save_path, "test")
    make_dir(save_path, train_path, test_path)
    generator(train_dataset, train_path, isShow=True)
    generator(test_dataset, test_path, isShow=False)
    print("生成完成！")
    # update



