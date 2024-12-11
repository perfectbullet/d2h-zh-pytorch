"""
@FileName：linear_regression_scratch.py
@Description：
@Author：zhoujing
@contact：121531845@qq.com
@Time：2024/12/7 11:01
@Department：红石扩大小区
@Website：www.zhoujing.com
@Copyright：©2019-2024 GX信息科技有限公司
"""
import torch
# from d2l import torch as d2l
from matplotlib import pyplot as plt
import random


print(torch.cuda.is_available())
true_w = torch.tensor([2, 3.4])
b = 4.2
num_examples = 1000
# def synthetic_data(w, b, num_examples):
X = torch.normal(0, 1, (num_examples, len(true_w)))
print(X[:1])
y = torch.matmul(X, true_w) + b
print(y[:1])
y = y + torch.normal(0, 0.01, y.shape)
print(y[:1])
# d2l.set_figsize()
print(X[:, 1])
# plt.scatter(X[:, 1].detach().numpy(), y.detach().numpy(), 1)


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    indices = random.shuffle(indices)

batch_size = 10

data_iter(batch_size, X, y)