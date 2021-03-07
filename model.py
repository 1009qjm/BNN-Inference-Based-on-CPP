from quantization import *

class Net(nn.Module):
    def __init__(self, cfg=None, A=2, W=2):
        super(Net, self).__init__()
        # 模型结构与搭建
        if cfg is None:
            cfg = [16, 32, 64, 10]
        self.tnn_bin = nn.Sequential(
            nn.Conv2d(1, cfg[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(cfg[0]),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Tnn_Bin_Conv2d(cfg[0], cfg[1], kernel_size=5, stride=1, padding=2, A=A, W=W),
            Tnn_Bin_Conv2d(cfg[1], cfg[1], kernel_size=5, stride=1, padding=2, A=A, W=W),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Tnn_Bin_Conv2d(cfg[1], cfg[2], kernel_size=5, stride=1, padding=2, A=A, W=W),
            Tnn_Bin_Conv2d(cfg[2], cfg[3], kernel_size=5, stride=1, padding=2, last_relu=1, A=A, W=W),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.tnn_bin(x)
        x = x.view(x.size(0), -1)
        return x

