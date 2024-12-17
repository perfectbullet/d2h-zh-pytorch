"""
@FileName：mlp-concise.py
@Description：
@Author：zhoujing
@contact：121531845@qq.com
@Time：2024/12/12 14:18
@Department：红石扩大小区
@Website：www.zhoujing.com
@Copyright：©2019-2024 GX信息科技有限公司
"""

import torch
from torch import nn

from utils import load_data_fashion_mnist, train_ch3, predict_ch3

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
loss = nn.CrossEntropyLoss(reduction='none')

sgd = torch.optim.SGD(net.parameters(), lr=0.1)

train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size=64)

train_ch3(net, train_dataloader, test_dataloader, loss, 10, sgd)

predict_ch3(net, test_dataloader)
