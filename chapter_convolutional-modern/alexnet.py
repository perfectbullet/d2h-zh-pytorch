import torch
from torch import nn

from utils import load_data_fashion_mnist, train_ch6, try_gpu

net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)

# torch.rand 均匀分布 [0, 1)
x = torch.rand((1, 1, 224, 224))
for layer in net:
    x = layer(x)
    print('{}\t{}'.format(layer.__class__.__name__, x.shape))

if __name__ == '__main__':
    # 读取数据
    train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size=64, resize=(224, 224))
    # 模型训练
    # net.to('cuda')
    # print(evaluate_accuracy_gpu(net, test_dataloader))
    # 输出0.1
    lr, num_epochs = 0.5, 20
    train_ch6(net, train_dataloader, test_dataloader, num_epochs, lr, try_gpu())
