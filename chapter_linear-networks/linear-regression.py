import random
import torch
from matplotlib import pyplot as plt

# 1.生成数据集
X = torch.normal(0, 1, (1000, 2))
features = X
print(X.shape)
print(X[:2])
w = torch.tensor([2, -3.4])
b = 4.2
y = torch.matmul(X, w) + b
y += torch.normal(0, 0.001, (1000, ))
labels = y.reshape((-1, 1))
# print(y.shape)
# print(y[:2])

plt.scatter(X[:, 0].detach().numpy(), y.detach().numpy(), 1)
plt.show()
# 2.读取数据集


def data_iter(features, labels, batch_size):
    n = len(labels)
    indexes = list(range(0, n))
    random.shuffle(indexes)
    for i in range(0, n, batch_size):
        batch_indexes = indexes[i: min(i+batch_size, n)]
        # print(batch_indexes)
        yield features[batch_indexes], labels[batch_indexes]

batch_size = 10
#
# for x, y in data_iter(X, y, batch_size):
#     print(x)
#     print(y)
#     break

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y, y_hat):
    return (y_hat - y) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 30
net = linreg
loss = squared_loss

# 训练
for epoch in range(num_epochs):
    for x_i, y_i in data_iter(features, labels, batch_size):
        l = loss(net(x_i, w, b), y_i)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    print('w: {}'.format(w))
    print('b: {}'.format(b))