# -*- coding: utf-8 -*-
# @Time    : 2021/4/8 10:58
# @Author  : kaka

import torch.nn as nn


class TextRNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 class_num,
                 embed_size=128,
                 hidden_size=128,
                 num_layers=1,
                 bidirectional=False,
                 dropout_rate=0.5):
        """
        :param vocab_size:     词典大小
        :param class_num:      类别数量
        :param embed_size:     词向量大小
        :param hidden_size:    RNN隐层大小
        :param num_layers:     RNN堆叠层数
        :param bidirectional:  是否双向连接
        :param dropout_rate:   全连接层dropout rate
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        hidden_num = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(hidden_num, class_num)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        :param x: shape [batch_size, seq_len]
        :return:
        """
        x = self.embed(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # get last step output
        out = self.fc(self.dropout(x))
        return out
