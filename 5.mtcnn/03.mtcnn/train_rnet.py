import net
import trainer
import os

if __name__ == '__main__':
    net = net.RNet()
    index = 28
    last_acc = 0.9
    last_r2 = 0.7
    save_path = fr"./params_r/r_net_{index}.pth"
    img_dir = r"F:\2.Dataset\mtcnn_12_m"

    if not os.path.exists("./params_r"):
        os.makedirs("./params_r")
    trainer = trainer.Trainer(net, save_path, img_dir, index, 24, last_acc, last_r2)
    trainer.train(0.6)
