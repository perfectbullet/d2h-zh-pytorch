"""
@FileName：parameters.py
@Description：
@Author：zhoujing
@contact：121531845@qq.com
@Time：2024/12/13 23:16
@Department：红石扩大小区
@Website：www.zhoujing.com
@Copyright：©2019-2024 GX信息科技有限公司
"""
# 参数绑定
# 有时我们希望在多个层间共享参数： 我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。
import torch
from torch import nn

share = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), share, nn.ReLU(), share, nn.Linear(8, 1))
x = torch.normal(0, 1, (10, 16))
out = net(x)
print(x)
