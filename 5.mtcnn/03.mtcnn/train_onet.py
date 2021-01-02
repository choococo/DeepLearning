import net
import train
import os

if __name__ == '__main__':
    net = net.ONet()
    index = 0
    save_path = fr"./params_o/o_net_{index}.pth"
    img_dir = r"E:\Dataset\mtcnn_dataset\test01"
    if not os.path.exists("./params_o"):
        os.makedirs("./params_o")
    trainer = train.Trainer(net, save_path, img_dir, index, 48, is_landmark=True)
    trainer.train(0.5)











