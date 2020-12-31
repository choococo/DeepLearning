import nets
import train
import os

if __name__ == '__main__':
    net = nets.PNet()
    if not os.path.exists("./param"):
        os.makedirs("./param")
    trainer = train.Trainer(net, './param/p_net.pth', r"F:\2.Dataset\mtcnn_dataset\testing\test01/train/12")
    trainer.train(0.7)
