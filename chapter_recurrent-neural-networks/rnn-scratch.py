import math
import torch
from torch import nn
from torch.nn import functional as F
import utils

batch_size = 32
num_steps = 35
train_iter, vocab = utils.load_data_time_machine(batch_size, num_steps)

# 独热编码
t = F.one_hot(torch.tensor([0, 2]), len(vocab))
#  我们经常转换输入的维度，以便获得形状为 （时间步数，批量大小，词表大小）的输出
X = torch.arange(10).reshape((2, 5))
t2 = F.one_hot(X.T, 28).shape
print(t2)


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

state = net.begin_state(X.shape[0], utils.try_gpu())
print('init state:', state)
Y, new_state = net(X.to(utils.try_gpu()), state)
print(Y.shape, len(new_state), new_state[0].shape)
# 我们可以看到输出形状是（时间步数批量大小，词表大小）
# 而隐状态形状保持不变，即（批量大小，隐藏单元数）


# 让我们首先定义预测函数来生成prefix之后的新字符， 其中的prefix是一个用户提供的包含多个字符的字符串。
# 在循环遍历prefix中的开始字符时， 我们不断地将隐状态传递到下一个时间步，但是不生成任何输出。
# 这被称为预热（warm-up）期， 因为在此期间模型会自我更新（例如，更新隐状态）， 但不会进行预测。
# 预热期结束后，隐状态的值通常比刚开始的初始值更适合预测， 从而预测字符并输出它们。
# def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
#     """在prefix后面生成新字符"""
#     state = net.begin_state(batch_size=1, device=device)
#     outputs = [vocab[prefix[0]]]
#     get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
#     for y in prefix[1:]:  # 预热期
#         _, state = net(get_input(), state)
#         outputs.append(vocab[y])
#     for _ in range(num_preds):  # 预测num_preds步
#         y, state = net(get_input(), state)
#         outputs.append(int(y.argmax(dim=1).reshape(1)))
#     return ''.join([vocab.idx_to_token[i] for i in outputs])

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

t = predict_ch8('time traveller ', 10, net, vocab, utils.try_gpu())
print(t)
