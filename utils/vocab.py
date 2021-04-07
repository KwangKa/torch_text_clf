# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 18:27
# @Author  : kaka


class Vocab(object):
    def __init__(self,
                 counter,
                 min_freq=1,
                 max_size=None):
        """
        :param counter: collections.Counter object holding the frequencies of each token found in the corpus
        :param min_freq: The minimum frequency needed to include a token in the vocabulary. Values less than 1 will be
                         set to 1. Default: 1
        :param max_size: The maximum size of the vocabulary, or None for no maximum. Default: None.
        """
        self.freqs = counter
        counter = counter.copy()
        special_tokens = ['<pad>', '<unk>']
        self.unk_index = 1
        for token in special_tokens:
            del counter[token]
        min_freq = max(min_freq, 1)
        max_size = None if max_size is None else max_size + len(special_tokens)

        self.itos = list(special_tokens)
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        self.stoi = {word: idx for idx, word in enumerate(self.itos)}

    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_index)

    def __len__(self):
        return len(self.itos)

    def tokens_to_ids(self, tokens):
        ids = [self.__getitem__(token) for token in tokens]
        return ids

    def ids_to_tokens(self, ids):
        tokens = [self.itos[idx] for idx in ids]
        return tokens
