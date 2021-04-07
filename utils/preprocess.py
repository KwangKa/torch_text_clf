# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 17:49
# @Author  : kaka

import json
from collections import Counter
from tqdm import tqdm
from .vocab import Vocab


def build_vocab(corpus_files, tokenizer, min_freq=1, max_size=None):
    counter = Counter()
    for fname in corpus_files:
        # print('processing file:{0}'.format(fname))
        with open(fname, 'r', encoding='utf8') as h_in:
            for line in tqdm(h_in):
                data = json.loads(line)
                text = data.get('text', '')
                tokens = tokenizer(text)
                counter.update(tokens)
    vocab = Vocab(counter, min_freq=min_freq, max_size=max_size)
    return vocab
