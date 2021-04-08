# -*- coding: utf-8 -*-
# @Time    : 2021/4/8 14:48
# @Author  : kaka

import torch
import torch.nn as nn


class TransformerClf(nn.Module):
    def __init__(self,
                 vocab_size,
                 class_num,
                 max_seq_length,
                 embed_size=128,
                 nhead=4,
                 num_layers=4,
                 dim_feedforward=2048,
                 dropout_rate=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_seq_length, embed_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_size)
        )
        self.fc = nn.Linear(embed_size, class_num)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        :param x: shape [batch_size, seq_len]
        :return:
        """
        x_word_embed = self.embed(x)
        x_pos_embed = self.pos_embed(torch.arange(0, x.size(1), dtype=torch.long).unsqueeze(0).repeat(x.size(0), 1))
        x = x_word_embed + x_pos_embed
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.mean(axis=0)  # 取encoder输出的向量平均
        output = self.fc(self.dropout(x))
        return output
