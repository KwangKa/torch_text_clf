# -*- coding: utf-8 -*-
# @Time    : 2021/4/7 16:01
# @Author  : kaka

import jieba


class Tokenizer(object):
    def __init__(self, f_stopwords=None):
        self.stopwords = set()
        if f_stopwords:
            stopwords = open(f_stopwords, 'r', encoding='utf8').read().splitlines()
            self.stopwords = set([w.lower() for w in stopwords])

    def tokenize(self, text):
        tokens = jieba.lcut(text)
        tokens = [w.lower() for w in tokens if w.strip() and w.lower() not in self.stopwords]
        return tokens
