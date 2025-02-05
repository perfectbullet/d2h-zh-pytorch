import math
import torch
from torch import nn
from torch.nn import functional as F
import utils

batch_size = 32
num_steps = 64
train_iter, vocab = utils.load_data_time_machine(batch_size, num_steps)

# 独热编码
# t = F.one_hot(torch.tensor([0, 2]), len(vocab))
# #  我们经常转换输入的维度，以便获得形状为 （时间步数，批量大小，词表大小）的输出
# X = torch.arange(10).reshape((2, 5))
# t2 = F.one_hot(X.T, 28).shape
# print(t2)


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度, 权重需要计算提督
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# 初始化
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


# 在一个时间步内计算隐状态和输出，
def rnn(inputs, state, params):
    # inputs 形状: (时间步数量，批量大小， 字典大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X 形状： （批量大小，字典大小）
    for X in inputs:
        # torch.mm 处理二维矩阵的乘法，而且也只能处理二维矩阵
        # 原本 H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        # Y 形状： （批量大小，字典大小）
        Y = torch.matmul(H, W_hq) + b_q
        # 按时间一个个放到outputs中
        outputs.append(Y)
    Y_out = torch.cat(outputs, dim=0)
    return Y_out, (H,)


class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
        print('device:', device)

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

# 测试输出是否正确
num_hiddens = 512
net = RNNModelScratch(
    len(vocab),
    num_hiddens,
    device=utils.try_gpu(),
    get_params=get_params,
    init_state=init_rnn_state,
    forward_fn=rnn
)

# state = net.begin_state(X.shape[0], utils.try_gpu())
# print('init state:', state)
# Y, new_state = net(X.to(utils.try_gpu()), state)
# print(Y.shape, len(new_state), new_state[0].shape)
# 我们可以看到输出形状是（时间步数批量大小，词表大小）
# 而隐状态形状保持不变，即（批量大小，隐藏单元数）


# Copy to clipboard
def predict_ch8(prefix, num_preds, net, vocab, device):
    # 在prefix后面生成新字符
    state = net.begin_state(batch_size=1, device=device)
    # outputs 是字母的索引列表
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 预热期
    for y in prefix[1:]:
        input_a = get_input()
        y_no_use, state = net(input_a, state)
        outputs.append(vocab[y])
    # 预测num_preds步
    for _ in range(num_preds):
        input_b = get_input()
        y_pred, state = net(input_b, state)
        outputs.append(int(y_pred.argmax(dim=1).reshape(1)))
    pred_list = [vocab.idx_to_token[i] for i in outputs]
    return ''.join(pred_list)

def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, utils.Timer()
    metric = utils.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        # else:
        #     if isinstance(net, nn.Module) and not isinstance(state, tuple):
        #         # state对于nn.GRU是个张量
        #         state.detach_()
        #     else:
        #         # state对于nn.LSTM或对于我们从零开始实现的模型是个元组
        #         for s in state:
        #             s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(net, nn.Module):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """ 训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = utils.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        optimizer = torch.optim.SGD(net.parameters(), lr)
    else:
        optimizer = lambda batch_size: utils.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(1, num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, optimizer, device, use_random_iter)
        if epoch % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

# 训练循环神经网络模型
num_epoches = 500
lr = 1
train_ch8(net, train_iter, vocab, lr, num_epoches, utils.try_gpu(0))
