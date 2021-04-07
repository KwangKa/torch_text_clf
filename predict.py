# -*- coding: utf-8 -*-
# @Time    : 2021/2/22 11:22
# @Author  : kaka

import json
import pickle
import torch
from model.textcnn import TextCNN
from utils.tokenizer import Tokenizer


def load_vocab(fname):
    with open(fname, 'rb') as h:
        vocab = pickle.load(h)
    return vocab


def load_model(f_model, f_vocab, f_conf):
    with open(f_conf, 'r') as h_in:
        args = json.load(h_in)

    vocab = load_vocab(f_vocab)
    vocab_size = len(vocab)

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


def to_tensor(sentence, tokenizer, vocab):
    tokens = tokenizer(sentence)
    index = vocab.tokens_to_ids(tokens)
    tensor = torch.LongTensor(index)
    tensor = tensor.unsqueeze(1).T
    return tensor


def main():
    model, vocab = load_model('./step300_acc90.8000.pt', './vocab.pkl', './conf/textcnn_conf.json')
    model.eval()

    tokenizer = Tokenizer().tokenize

    class_name = ['entertainment', 'sports', 'finance']
    sentences = [u'火箭队史最佳阵容，姚明大梦制霸内线，哈登麦迪完爆勇士水花兄弟',
                 u'平安好医生并不孤单 细数那些从破发开始星辰大海征途的伟大公司',
                 u'如果现在由你来接任中国足协主席，你会怎么样做才能提高中国足球整体水平？',
                 u'吴广超：5.8伦敦金关注1325争夺继续空，原油择机中空',
                 u'西仪股份等5只军工股涨停 机构：业绩有望超预期',
                 u'刘涛：出席活动！网友：我只看到她的一条腿！',
                 u'忘尽心中情，刘德华版《苏乞儿》的主题曲，老歌经典豪气']
    for sent in sentences:
        t = to_tensor(sent, tokenizer, vocab)
        pred_class = torch.argmax(model(t).squeeze(0)).item()
        print(u'{0}\t预测类别:{1}'.format(sent, class_name[pred_class]))


if __name__ == '__main__':
    main()
