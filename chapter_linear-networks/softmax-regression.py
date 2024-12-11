from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from triton.interpreter.memory_map import torch

from utils import set_axes, Accumulator, accuracy, evaluate_accuracy


def load_data_fashion_mnist(batch_size = 64):
    # Create data loaders.

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader

batch_size = 64
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
for X, y in test_iter:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
# 初始化模型参数
num_inputs = 784
num_outputs = 10
w = torch.normal(0, 1, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
print('w shape {}'.format(w.shape))
print('b shape {}'.format(b.shape))

# 定义softmax操作
def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(axis=1, keepdims=True)
    return x_exp / partition

# 定义模型
def net(x):
    re_x = x.reshape(-1, w.shape[0])
    output = torch.matmul(re_x, w) + b
    return softmax(output)


# 定义损失函数
def cross_entropy(y_hat, y):
    # y 是正确分类的标号 (784, 1)
    # y_hat 是一个概率的tensor shape (784, 10)
    selected_y_row = list(range(0, y.shape[0]))
    # y_hat 所有行的指定下标
    selected_y_hat_prob = y_hat[selected_y_row, y]
    return -torch.log(selected_y_hat_prob)

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train_epoch_ch3(net, train_iter, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(x)
        l = cross_entropy(y_hat, y)
        # 计算梯度并更新参数
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater([w, b], lr, batch_size)
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, num_epochs):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, sgd)
        test_acc = evaluate_accuracy(net, test_iter)
        train_loss, train_acc = train_metrics
        print('train_loss: {}, train_acc: {}, test_acc: {}'.format(train_loss, train_acc, test_acc))


if __name__ == '__main__':

    lr = 0.1
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, num_epochs)
