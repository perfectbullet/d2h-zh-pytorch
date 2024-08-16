import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256  # batch_size 设为256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 仍使用 Fashion-MNIST 数据集


X = torch.tensor([[1., 2., 3.], [4., 5.,6.]])
X.sum(0, keepdim=True), X.sum(1, keepdim=True)  # 非降维求和 参见2.3.6
