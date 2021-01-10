import net
import trainer
import os

if __name__ == '__main__':
    net = net.PNet()
    index = 30
    last_acc = 0.9803
    last_r2 = 0.8545
    save_path = fr"./params_p/p_net_{index}.pth"
    img_dir = r"F:\Dataset02\mtcnn_fxs"
    if not os.path.exists("./params_p"):
        os.makedirs("./params_p")
    trainer = trainer.Trainer(net, save_path, img_dir, index, 12, last_acc, last_r2)
    trainer.train(0.7)











