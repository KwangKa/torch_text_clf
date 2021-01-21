# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 10:26
# @Author  : kaka

import jieba
import torchtext.data as tdata


def jieba_cut(text):
    return [w for w in jieba.lcut(text) if w.strip()]


def get_data(path, batch_size, device):
    LABEL = tdata.Field(sequential=False, use_vocab=False)
    TEXT = tdata.Field(sequential=True, tokenize=jieba_cut, use_vocab=True)
    train, val, test = tdata.TabularDataset.splits(
        path=path,
        train='train.json',
        validation='val.json',
        test='test.json',
        format='json',
        fields={
            'text': ('text', TEXT),
            'category_id': ('label', LABEL)
        }
    )

    TEXT.build_vocab(train, min_freq=3)

    train_iter = tdata.Iterator(
        dataset=train,
        batch_size=batch_size,
        train=True,
        device=device
    )

    val_iter = tdata.Iterator(
        dataset=val,
        batch_size=batch_size,
        train=True,
        device=device
    )

    test_iter = tdata.Iterator(
        dataset=test,
        batch_size=batch_size,
        train=False,
        sort=False,
        device=device
    )

    return train_iter, val_iter, test_iter
