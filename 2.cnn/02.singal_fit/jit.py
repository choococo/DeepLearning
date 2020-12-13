import torch
from net import Net
from torch import jit

if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load("params/19.t"))

    # 模拟一个输入（占位）
    input = torch.randn(1, 1, 9)
    torch_model = jit.trace(model, input) # 打包对象

    torch_model.save("waveform.pt") # 一定要是pt文件

