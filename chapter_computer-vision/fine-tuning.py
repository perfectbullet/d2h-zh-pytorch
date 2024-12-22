import torch
import os
import torchvision
from torch import nn
import utils

# 获取数据集
data_dir = utils.download_extract('hotdog')

train_images = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_images = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

hotdogs = [train_images[i][0] for i in range(8)]
not_hotdogs = [train_images[-i - 1][0] for i in range(8)]
# utils.show_images(hotdogs + not_hotdogs, 2, 8, scale=2)

normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# 在训练期间，我们首先从图像中裁切随机大小和随机长宽比的区域
# 然后将该区域缩放为224*224输入图像
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

# 定义和初始化模型¶
# pretrained_net = torchvision.models.resnet18()
# print(pretrained_net.fc)


# 本地加载模型
model_path = '../resnet18-f37072fd.pth'
# 使用 torch.hub.load 加载模型
# pretrained_net = torchvision.models.resnet18()
# pretrained_net.load_state_dict(torch.load(model_path))
# print(pretrained_net.fc)
# for layer in pretrained_net:
#     print(layer)

finetune_net = torchvision.models.resnet18()
finetune_net.load_state_dict(torch.load(model_path))
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)
# print(finetune_net)
print(finetune_net.fc)


# 微调模型
# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = utils.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        # 非 fc 层的参数
        params_1x = [param for name, param in net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
        params_fc = net.fc.parameters()
        trainer = torch.optim.SGD([
            {'params': params_1x},
            {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
            lr=learning_rate,
            weight_decay=0.001
        )
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    utils.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=devices)

if __name__ == '__main__':

    train_fine_tuning(finetune_net, 5e-5)
