import torch
from logger.loggers import classLogger, funcLogger
from TrainTemp.net import NetTempV1


@classLogger
class Package(object):

    def __init__(self, params_path=None, save_params_name=None):
        self.model = NetTempV1()
        self.model.load_state_dict(torch.load(params_path))
        self.input = torch.randn(1, 784)
        self.save_params_name = save_params_name

    @funcLogger
    def __call__(self):
        traced_script_model = torch.jit.trace(self.model, self.input)
        traced_script_model.save(f"{self.save_params_name}.pt")


if __name__ == '__main__':
    model = NetTempV1()
    model.load_state_dict(torch.load("./params/9.t"))

    # 模拟一个输入
    input = torch.randn(1, 784)

    # 打包
    traced_script_moudle = torch.jit.trace(model, input)
    traced_script_moudle.save("mnist.pt")
