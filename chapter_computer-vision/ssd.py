import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import utils

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(
        num_inputs,
        num_anchors * (num_classes + 1),
        kernel_size=3,
        padding=1
    )

def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(
        num_inputs,
        num_anchors * 4,
        kernel_size=3,
        padding=1
    )

# 连结多尺度的预测
def forward(x, block):
    return block(x)

Y1 = forward(torch.rand(2, 8, 20, 20), cls_predictor(8, 5, 10))
Y2 = forward(torch.rand(2, 16, 10, 10), cls_predictor(16, 3, 10))
print(Y1.shape, Y2.shape)

def flatten_pred(pred):

