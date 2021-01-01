import net
import train
import os

if __name__ == '__main__':
    net = net.PNet()
    net.parameters()
    index = 0
    save_path = fr"./params_p/p_net_{index}.pth"
    img_dir = r"F:\2.Dataset\mtcnn_dataset\testing\test01"
    if not os.path.exists("./params_p"):
        os.makedirs("./params_p")
    trainer = train.Trainer(net, save_path, img_dir, 48)
    trainer.train(0.7)










