import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import utils


def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)


# 边框预测层
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


# 连接多尺度预测
def forward(x, block):
    return block(x)


# Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
# Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
# print(Y1.shape, Y2.shape)


def flatten_pred(pred):
    # 把通道移动到最后一维
    t1 = pred.permute(0, 2, 3, 1)
    # 除了批量，其他维度拉平
    t2 = torch.flatten(t1, start_dim=1)
    print('pred shape ', t2.shape)
    return t2


def concat_preds(preds):
    t1 = [flatten_pred(p) for p in preds]
    # 在同一个小批量的两个不同尺度上连接
    # 第一个 pred shape  torch.Size([2, 22000])
    # 第二个 pred shape  torch.Size([2, 3300])
    # 结果 torch.Size([2, 25300])
    t2 = torch.concat(t1, dim=1)
    return t2


# t3 = concat_preds([Y1, Y2])
# print(t3.shape)


# 高宽减半块
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        #  对于所有的batch中样本的同一个channel的数据元素进行标准化处理，即如果有C个通道，
        #  无论batch中有多少个样本，都会在通道维度上进行标准化处理，一共进行C次。
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    # 不指定 stride 情况下，步幅与kernel size大小相同
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


# print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)


# 基本网络块
def base_net():
    # 我们构造了一个小的基础网络，该网络串联3个高和宽减半块，并逐步将通道数翻倍。
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


# print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)

# ##### 完整的网络

# 完整的单发多框检测模型由五个模块组成。每个块生成的特征图既用于生成锚框，又用于预测这些锚框的类别和偏移量。
def get_blk(i):
    if i == 0:
        # 第一个是基本网络块
        blk = base_net()
    elif i == 1:
        # 第二个到第四个是高和宽减半块
        blk = down_sample_blk(64, 128)
    elif i == 4:
        #
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        # i = 3, i = 2
        # 最后一个模块使用全局最大池将高度和宽度都降到1
        blk = down_sample_blk(128, 128)
    return blk


# 每个块定义前向传播。
# 与图像分类任务不同，此处的输出包括：1.CNN特征图Y；2。在当前尺度下根据Y生成的锚框；3.预测的这些锚框的类别和偏移量（基于Y）
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    # 生成 anchors
    anchors = utils.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return Y, anchors, cls_preds, bbox_preds
