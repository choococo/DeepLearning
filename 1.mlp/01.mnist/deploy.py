from torch import jit
from net import Net
import torch

if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load("./params/9.t"))

    # 模拟一个输入
    input = torch.randn(1, 784)

    # 打包
    traced_script_moudle = torch.jit.trace(model, input)
    traced_script_moudle.save("mnist.pt")
