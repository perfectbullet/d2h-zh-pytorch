import torch
from torch import nn
# F.relu()是函数调用，一般使用在foreward函数里。而nn.ReLU()是模块调用，一般在定义网络层的时候使用。
from torch.nn import functional as F
import utils


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            # 高宽减半的时候，也就是 strides 为 2 的时候，要使用 1*1 的核融合通道，并把x输入的高宽减半
            # 这样才可以做残差链接
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # batch norm 不改变通道数和高宽
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


# X = torch.rand(4, 3, 6, 6)
# blk = Residual(3, 6, use_1x1conv=True, strides=2)
# print(blk(X).shape)
# blk = Residual(3,6, use_1x1conv=True, strides=2)
# blk(X).shape

b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)


# ResNet则使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(0, num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            # 每个block的第二个Residual不修改高宽和channel
            blk.append(Residual(num_channels, num_channels))
    return blk


# 在ResNet加入所有残差块，这里每个模块使用2个残差块
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))


# 自适应平均池化，指定输出（H，W） -> (1, 1)
net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 10)
)

#
# x = torch.rand(size=(1, 1, 224, 224))
# for layer in net:
#     x = layer(x)
#     print(layer.__class__.__name__, 'out shape \t', x.shape)
if __name__ == '__main__':

    lr = 0.1
    num_epochs = 10
    batch_size = 256
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=96)
    utils.train_ch6(net, train_iter, test_iter, lr=lr, num_epochs=num_epochs, device=utils.try_gpu())
