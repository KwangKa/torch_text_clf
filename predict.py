# -*- coding: utf-8 -*-
# @Time    : 2021/2/22 11:22
# @Author  : kaka

import json
import pickle
import torch
from dataset import jieba_cut
from model.textcnn import TextCNN


def load_vocab(fname):
    with open(fname, 'rb') as h:
        vocab = pickle.load(h)
    return vocab


def load_model(f_model, f_vocab, f_conf):
    with open(f_conf, 'r') as h_in:
        args = json.load(h_in)

    vocab = load_vocab(f_vocab)
    vocab_size = len(vocab.stoi)
    # print('vocab size:{0}'.format(vocab_size))

    model = TextCNN(
        vocab_size=vocab_size,
        class_num=args['class_num'],
        embed_size=args['embed_size'],
        kernel_num=args['kernel_num'],
        kernel_sizes=args['kernel_sizes'],
        dropout_rate=args['dropout_rate']
    )
    model.load_state_dict(torch.load(f_model))
    return model, vocab


def to_tensor(sentence, vocab):
    tokens = jieba_cut(sentence)
    index = [vocab.stoi[w] for w in tokens]
    tensor = torch.LongTensor(index)
    tensor = tensor.unsqueeze(1).T
    return tensor


def main():
    model, vocab = load_model('./step240_acc89.1000.pt', './vocab.pkl', './conf/textcnn_conf.json')
    model.eval()
    # print(model)

    sentences = [u'火箭队史最佳阵容，姚明大梦制霸内线，哈登麦迪完爆勇士水花兄弟',
                 u'平安好医生并不孤单 细数那些从破发开始星辰大海征途的伟大公司',
                 u'忘尽心中情，刘德华版《苏乞儿》的主题曲，老歌经典豪气']
    for sent in sentences:
        t = to_tensor(sent, vocab)
        pred_class = torch.argmax(model(t).squeeze(0)).item()
        print(u'{0}\t预测类别:{1}'.format(sent, pred_class))


if __name__ == '__main__':
    main()
