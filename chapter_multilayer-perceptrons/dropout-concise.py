"""
@FileName：dropout.py
@Description：
@Author：zhoujing
@contact：121531845@qq.com
@Time：2024/12/12 17:30
@Department：红石扩大小区
@Website：www.zhoujing.com
@Copyright：©2019-2024 GX信息科技有限公司
"""
import torch
from torch import nn
from utils import load_data_fashion_mnist, train_ch3, predict_ch3


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
    else:
        print(m.type)


if __name__ == '__main__':
    # 定义模型参数
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    num_epochs, lr, batch_size = 10, 0.1, 256
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(num_hiddens2, num_outputs)
    )
    net.apply(init_weight)

    train_loader, test_loader = load_data_fashion_mnist(batch_size=64)
    loss = nn.CrossEntropyLoss(reduction='none')
    sgd = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch3(net, train_loader, test_loader, loss, num_epochs, sgd)
    predict_ch3(net, test_loader)

