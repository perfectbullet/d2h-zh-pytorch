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


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    #  如上所述, 为了保持 h' 的期望不变, 重新缩放剩余部分：将剩余部分除以1.0-dropout
    return mask * X / (1.0 - dropout)

#
# X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
# print(X)
# print(dropout_layer(X, 0.))
# print(dropout_layer(X, 0.5))
# print(dropout_layer(X, 1.))




class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super().__init__()
        self.num_inputs = num_inputs
        self.linear1 = nn.Linear(num_inputs, num_hiddens1)
        self.linear2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.linear3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
        self.is_training = True

    def forward(self, x):
        h1 = self.relu(self.linear1(x.reshape(-1, self.num_inputs)))
        # 只在训练是使用dropout算法,
        if self.is_training:
            # 在第一个全连接层后使用 dropout
            dropout_layer(h1, 0.2)
        h2 = self.relu(self.linear2(h1))
        if self.is_training:
            # 在第一个全连接层后使用 dropout
            dropout_layer(h1, 0.5)
        out = self.linear3(h2)
        return out


if __name__ == '__main__':
    # 定义模型参数
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    num_epochs, lr, batch_size = 10, 0.1, 256
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    train_loader, test_loader = load_data_fashion_mnist(batch_size=64)
    loss = nn.CrossEntropyLoss(reduction='none')
    sgd = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch3(net, train_loader, test_loader, loss, num_epochs, sgd)
    predict_ch3(net, test_loader)

