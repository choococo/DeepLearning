import net
import trainer
import os

if __name__ == '__main__':
    net = net.ONet()
    index = 37
    last_acc = 0.9
    last_r2 = 0.8
    save_path = fr"./params_o/o_net_{index}.pth"
    img_dir = r"F:\2.Dataset\mtcnn_12_m"
    if not os.path.exists("./params_o"):
        os.makedirs("./params_o")
    trainer = trainer.Trainer(net, save_path, img_dir, index, 48, last_acc, last_r2)
    trainer.train(0.5)
