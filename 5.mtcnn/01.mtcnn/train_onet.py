import nets
import train
import os

if __name__ == '__main__':
    net = nets.ONet()
    if not os.path.exists("./param"):
        os.makedirs("./param")
    trainer = train.Trainer(net, './param/o_net.pth', r"E:\datasets\train\48")
    trainer.train(0.5)
