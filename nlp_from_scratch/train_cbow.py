# -*- encoding: utf-8 -*-
'''
@File    :   train_cbow.py
@Time    :   2021/11/20 21:14:17
@Author  :   YueDaoWenEr 
'''

import sys
sys.path.append('.')
sys.path.append('..')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


from preprocess import build_corpus, creat_contexts_target, to_one_hot
from cbow import SimpleCBOW
from optimizer import Adam, SGD
from trainer import Trainer


def train_CBOW():

    window_size = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    text = "You say goodbye and I say hi."
    corpus, word2id, id2word = build_corpus(text)
    contexts, target = creat_contexts_target(corpus, window_size=window_size)
    vocab_size = len(word2id)
    contexts = to_one_hot(contexts, vocab_size)
    target = to_one_hot(target, vocab_size)
  
    model = SimpleCBOW(vocab_size, hidden_size)
    optimizer = Adam()  # todo:怎么实现的？
    # todo: Adam和SGD的曲线区别好大。
    trainer = Trainer(model, optimizer)
    trainer.fit(contexts, target, max_epoch=max_epoch, batch_size=batch_size)
    trainer.plot()  # 振荡不收敛？

    pass


def main():
    train_CBOW()
    pass


if __name__ == "__main__":
    main()

