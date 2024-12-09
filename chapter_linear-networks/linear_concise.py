import torch
from sklearn.utils import shuffle


def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y

w1 = torch.tensor([2, -3.4])
b1 = 4.2
num_examples = 1000
features, labels = synthetic_data(w1, b1, num_examples)

# 读取数据集
from torch.utils import data

batch_size = 10
dataset = data.TensorDataset(features, labels)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
t = next(iter(dataloader))
# print(t)
# 定义模型
from torch import nn

net = nn.Sequential(nn.Linear(2, 1, bias=True))
net[0].weight.data.normal_(0, 1)
net[0].bias.data.fill_(0)

# 定义损失函数
# [计算均方误差使用的是MSELoss类，也称为平方范数]。 默认情况下，它返回所有样本损失的平均值。
loss = nn.MSELoss()
# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
for epoch in range(0, 10):
    for x, y in dataloader:
        y_hat = net(x)
        trainer.zero_grad()
        l = loss(y_hat.reshape(y.shape), y)
        l.backward()
        trainer.step()


    with torch.no_grad():
        y_hat = net(features)
        l = loss(y_hat.reshape(labels.shape), labels)
        print('loss is {}'.format(l))

w = net[0].weight.data
print('w的估计误差：', w1 - w.reshape(w.shape), w.reshape(w.shape))
b = net[0].bias.data
print('b的估计误差：', b1 - b, b)
