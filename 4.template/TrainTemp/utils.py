import matplotlib.pyplot as plt
from logger.loggers import funcLogger


@funcLogger
def draw_acc_loss_figure(index, epoch_list, train_acc_list, val_acc_list, train_loss_list, val_loss_list):
    # plt.ion()
    plt.subplot(121)
    plt.title("Accuracy")
    plt.plot(epoch_list, train_acc_list, c="orange", label="train")
    plt.plot(epoch_list, val_acc_list, label="val")
    plt.legend()

    plt.subplot(122)
    plt.title("Loss")
    plt.plot(epoch_list, train_loss_list, c="orange", label="train")
    plt.plot(epoch_list, val_loss_list, label="val")
    plt.legend()

    # plt.ioff()  # 关闭实时画图
    plt.savefig(f"{index}.jpg")
    plt.show()
    # plt.pause(0.1)
    # plt.ioff()
    # plt.close()