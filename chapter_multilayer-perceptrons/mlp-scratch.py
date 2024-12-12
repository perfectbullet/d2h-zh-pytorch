"""
@FileName：mlp-scratch.py
@Description：
@Author：zhoujing
@contact：121531845@qq.com
@Time：2024/12/11 22:45
@Department：红石扩大小区
@Website：www.zhoujing.com
@Copyright：©2019-2024 GX信息科技有限公司
"""

import torch
from torch import nn
from utils import load_data_fashion_mnist, evaluate_accuracy, train_epoch_ch3, train_ch3


def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)


def net(x: torch.Tensor):
    x = x.reshape(-1, num_input)
    a1 = torch.matmul(x, w1)
    a2 = a1 + b1
    h1 = relu(a2)
    out = torch.matmul(h1, w2) + b2
    return out


if __name__ == '__main__':

    batch_size = 64
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
    for x, y in train_iter:
        print(x.shape)
        break

    num_input = 784
    num_output = 10
    num_hiddens = 256

    w1 = nn.Parameter(torch.normal(0, 1, (num_input, num_hiddens), requires_grad=True))
    b1 = nn.Parameter(torch.ones(num_hiddens, requires_grad=True))

    w2 = nn.Parameter(torch.normal(0, 1, (num_hiddens, num_output), requires_grad=True))
    b2 = nn.Parameter(torch.ones(num_output, requires_grad=True))

    params = [w1, b1, w2, b2]

    # reduction (str, 可选): 指定应用于输出的归约方式。
    # 可选值为 'none'、'mean'、'sum'。
    # 'none' 表示不进行归约，'mean' 表示对所有样本的损失求平均，'sum' 表示对所有样本的损失求和。
    # 默认值是 'mean'。
    loss = nn.CrossEntropyLoss(reduction='none')
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
