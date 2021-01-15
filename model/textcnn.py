# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 13:23
# @Author  : kaka

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 class_num,
                 embed_size=128,
                 kernel_num=100,
                 kernel_sizes=(2, 3, 4),
                 dropout_rate=0.5):
        """
        :param vocab_size:     词典大小
        :param class_num:      类别数量
        :param embed_size:     词向量大小
        :param kernel_num:
        :param kernel_sizes:
        :param dropout_rate:
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            self.convs.append(nn.Conv1d(embed_size, kernel_num, ks))
        hidden_num = len(kernel_sizes) * kernel_num
        self.fc = nn.Linear(hidden_num, class_num)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        :param x: shape [batch_size, seq_len]
        :return:
        """
        x = self.embed(x)
        x = x.transpose(1, 2)
        conv_outs = []
        for conv in self.convs:
            c_out = F.relu(conv(x))
            conv_outs.append(torch.topk(c_out, 1)[0].squeeze(2))  # max pool over time
        out = torch.cat(conv_outs, 1)
        out = self.fc(self.dropout(out))
        return out
