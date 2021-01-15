# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 15:22
# @Author  : kaka

import torch
import argparse
import json
from model.textcnn import TextCNN

with open('./conf/textcnn_conf.json', 'r') as h_in:
    args = json.load(h_in)

model = TextCNN(
    vocab_size=args['vocab_size'],
    class_num=args['class_num'],
    embed_size=args['embed_size'],
    kernel_num=args['kernel_num'],
    kernel_sizes=args['kernel_sizes'],
    dropout_rate=args['dropout_rate']
)
print(model)

x = torch.randint(0, args['vocab_size'], (3, 10))
print(model(x))
