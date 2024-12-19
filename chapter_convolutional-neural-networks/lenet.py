import torch
from torch import nn
from utils import load_data_fashion_mnist, train_ch6, try_gpu

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=16 * 5 * 5, out_features=120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10),
)

# torch.rand 均匀分布 [0, 1)
x = torch.rand((1, 1, 28, 28))
for layer in net:
    x = layer(x)
    print('{}\t{}'.format(layer.__class__.__name__, x.shape))

if __name__ == '__main__':
    # 读取数据
    train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size=64)
    # 模型训练
    # net.to('cuda')
    # print(evaluate_accuracy_gpu(net, test_dataloader))
    # 输出0.1
    lr, num_epochs = 0.5, 20
    train_ch6(net, train_dataloader, test_dataloader, num_epochs, lr, try_gpu())
