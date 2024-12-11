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
from utils import load_data_fashion_mnist, evaluate_accuracy

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


def train_epoch(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数

    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])


def train_model(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""

    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print('test_acc is {}'.format(test_acc))


loss = nn.CrossEntropyLoss()


num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
train_model(net, train_iter, test_iter, loss, num_epochs, updater)
