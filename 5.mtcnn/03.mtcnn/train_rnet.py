import net
import train
import os

if __name__ == '__main__':
    net = net.RNet()
    index = 0
    save_path = fr"./params_r/r_net_{index}.pth"
    img_dir = r"E:\Dataset\mtcnn_dataset\test01"
    if not os.path.exists("./params_r"):
        os.makedirs("./params_r")
    trainer = train.Trainer(net, save_path, img_dir, 24)
    trainer.train(0.6)











