import net
import train
import os

if __name__ == '__main__':
    net = net.PNet()
    index = 31
    save_path = fr"./params_p/p_net_{index}.pth"
    img_dir = r"E:\Dataset\mtcnn_dataset\test01"
    if not os.path.exists("./params_p"):
        os.makedirs("./params_p")
    trainer = train.Trainer(net, save_path, img_dir, index, 12)
    trainer.train(0.7)











