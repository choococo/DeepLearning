from TrainTemp.train import Train

if __name__ == '__main__':
    BATCH_SIZE = 100
    EPOCH = 35
    root = r"F:\2.Dataset\cat_dog"
    index = 0
    train = Train(root=root, batch_size=BATCH_SIZE, index=index, epoch=EPOCH, is_train=False)
    train()