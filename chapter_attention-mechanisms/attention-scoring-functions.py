import math
import torch
from torch import nn
from d2l import torch as d2l


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        re_X = X.reshape(shape)
        softmax_X = nn.functional.softmax(re_X, dim=-1)
        return softmax_X


class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # nn.Linear  不改变 dimension
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries_new, keys_new = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        queries_new_us = queries_new.unsqueeze(2)
        keys_new_us = keys_new.unsqueeze(1)
        features = queries_new_us + keys_new_us
        # features_np = features.detach().numpy()
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。   self.w_v 是 加性球和后的结果，在输入单层 MLP 中，w_v 就是那个 MLP
        # scores 的形状：(batch_size，查询的个数，“键-值”对的个数), scores注意力分数， 他是 key 和 query 做加性注意力的产物
        scores_origin = self.w_v(features)
        scores = scores_origin.squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的长度)
        # self.attention_weights 的形状：(batch_size，查询的个数，“键-值”对的个数)
        attention_gather = torch.bmm(self.attention_weights, values)
        # attention_weights_np = self.attention_weights.detach().numpy()
        return attention_gather


if __name__ == '__main__':
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    attention.eval()
    attention(queries, keys, values, valid_lens)
    d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')
    d2l.plt.show()

