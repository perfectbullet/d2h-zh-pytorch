import torch

import os

import d2l.torch as d2l
from torch import nn


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
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为 0
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


def read_data_cmn():
    """载入“英语－中文”数据集"""
    cache_dir = os.path.join('..', 'data')
    data_dir = os.path.join(cache_dir, 'cmn-eng')

    with open(os.path.join(data_dir, 'cmn.txt'), 'r',
              encoding='utf-8') as f:
        return f.read()


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列

    Defined in :numref:`sec_machine_translation`"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def build_array_cmn(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量

    Defined in :numref:`subsec_mt_data_loading`"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len


def load_data_cmn(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表

    Defined in :numref:`subsec_mt_data_loading`"""
    text = preprocess_cmn(read_data_cmn())
    source, target = tokenize_cmn(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_cmn(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_cmn(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


def preprocess_cmn(text):
    """预处理“英语－法语”数据集"""

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_cmn(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口

    Defined in :numref:`sec_encoder-decoder`"""

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类

    Defined in :numref:`sec_encoder-decoder`"""

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class Seq2SeqEncoder(Encoder):
    """
    用于序列到序列学习的循环神经网络编码器
    Defined in :numref:`sec_seq2seq`
    """

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # 接收到 X (batch_size,num_steps,)

        # embedding 输出'X'的形状：(batch_size,num_steps,embed_size)
        # 每个 单词 embedding 为长度为  embed_size
        X = self.embedding(X)

        # 在循环神经网络模型中，第一个轴对应于时间步
        # 时间步简单理解为， 一个样本句子的单词个数，有些是单词个数
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""

    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # 返回的 outputs的形状为 (batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return outputs.permute(1, 0, 2), hidden_state, enc_valid_lens

    def forward(self, X, state):
        # 输入 X 形状是 (batch_size, num_steps)

        # enc_outputs的形状为 (batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # hidden_state[-1]
            last_hidden_state = hidden_state[-1]
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(last_hidden_state, dim=1)
            # context的形状为(batch_size,1,num_hiddens)
            context = self.attention(
                query,
                enc_outputs,
                enc_outputs,
                enc_valid_lens
            )

            # us_x 形状是 (batch_size, 1, num_steps)
            us_x = torch.unsqueeze(x, dim=1)
            # 在特征维度上连结
            x = torch.cat((context, us_x), dim=-1)
            # 然后， x.permute(1, 0, 2) 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # cat_out 形状是 (num_steps,batch_size,num_hiddens)
        cat_out = torch.cat(outputs, dim=0)

        # 全连接层变换后，shuoutputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(cat_out)
        # outputs.permute(1, 0, 2) 的形状为 (batch_size, um_steps, vocab_size)
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


if __name__ == '__main__':
    # print('ok')
    # raw_text = read_data_cmn()
    # # print(raw_text[:1000])
    # text = preprocess_nmt(raw_text)
    # print(text[:80])
    # source, target = tokenize_nmt(text)
    # print(source[:6], target[:6])

    # 使用包含7个时间步的4个序列输入的小批量测试Bahdanau注意力解码器。
    # encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    # eval() Sets the module in evaluation mode.
    # encoder.eval()
    # decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    # decoder.eval()
    # X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
    # encoder_out = encoder(X)
    # state = decoder.init_state(encoder_out, None)

    # output, state = decoder(X, state)
    # print('##########', output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape)

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = load_data_cmn(batch_size, num_steps)
    encoder = d2l.Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
