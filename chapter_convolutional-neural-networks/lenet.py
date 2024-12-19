import torch
from torch import nn
from utils import load_data_fashion_mnist, Accumulator, accuracy, Animator, Timer

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=16*5*5, out_features=120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10),
)

# torch.rand 均匀分布 [0, 1)
x = torch.rand((1, 1, 28, 28))
for layer in net:
    x = layer(x)
    print('{}\t{}'.format(layer.__class__.__name__, x.shape))

def evaluate_accuracy_gpu(net, test_loader, device=None):
    '''
    使用gpu计算精度
    '''
    if isinstance(net, nn.Module):
        # 设置评估模式
        net.eval()
        if device is None:
            device = next(iter(net.parameters())).device
    print('evaluate on ', device)
    # 正确预测数量， 总数量
    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in test_loader:
            if isinstance(x, list):
                # bert微调所需
                x = [xi.to(device) for xi in x]
            else:
                x = x.to(device)
            y = y.to(device)
            metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]



def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def init_weights(m):
    """初始化权重 xavier_uniform_ """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用gpu训练模型"""
    net.apply(init_weights)
    print('trainning on ', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # reduction='mean' 取loss输出的加权平均值
    loss = nn.CrossEntropyLoss(reduction='mean')
    animator = Animator(
        xlabel='epoch',
        xlim=[1, num_epochs],
        ylim=[0.3, 0.9],
        legend=['train loss', 'train acc', 'test acc']
    )
    timer = Timer()
    avg_train_loss = None
    train_accuracy = None
    test_acc = None
    metric = None
    for epoch in range(1, num_epochs + 1):
        # loss, train accuracy, test accuracy
        metric = Accumulator(3)
        timer.start()
        net.train()
        avg_train_loss = None
        train_accuracy = None
        for X, y in train_iter:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), y.numel())
            avg_train_loss = metric[0] / metric[2]
            train_accuracy = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch, (avg_train_loss, train_accuracy, test_acc))
        timer.stop()
        print('epoch {}, train_loss {}, train_acc is {}'.format(epoch, avg_train_loss, train_accuracy))
    print(f'loss {avg_train_loss:.3f}, train acc {train_accuracy:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')

if __name__ == '__main__':
    # 读取数据
    train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size=64)
    # 模型训练
    # net.to('cuda')
    # print(evaluate_accuracy_gpu(net, test_dataloader))
    # 输出0.1
    lr, num_epochs = 0.5, 20
    train_ch6(net, train_dataloader, test_dataloader, num_epochs, lr, try_gpu())
