# -*- coding: utf-8 -*-
"""
@author: Me
@time: 2021/5/27 11:12
@description: 参考： https://blog.csdn.net/weixin_40699243/article/details/109271365
"""
import jieba
import keras.backend as K
import numpy as np
from keras.layers import Input, Embedding, Lambda
from keras.models import Model

#
word_size = 12  # 词向量维度
window = 3  # 窗口大小
nb_negative = 2  # 随机负采样的样本数
min_count = 1  # 频数少于min_count的词将会被抛弃，低频词类似于噪声，可以抛弃掉
nb_epoch = 2  # 迭代次数


def get_corpus(file):
    words = []  # 词库，不去重
    corpus = []
    try:
        with open(file, encoding='gbk') as fr:
            for line in fr:
                words += jieba.lcut(line)  # jieba分词，将句子切分为一个个词，并添加到词库中
                corpus.append(jieba.lcut(line))
    except:
        pass
    return words, corpus


def word2id(words):
    total = sum(words.values())  # 总词频
    words = {i: j for i, j in words.items() if j >= min_count}  # 去掉低频词
    id2word = {i + 2: j for i, j in enumerate(words)}  # id到词语的映射，习惯将0设置为PAD，1设置为UNK
    id2word[0] = 'PAD'
    id2word[1] = 'UNK'
    word2id = {j: i for i, j in id2word.items()}  # 词语到id的映射
    nb_word = len(id2word)  # 总词数
    print('id2word', id2word, 'word2id', word2id, 'nb_word', nb_word)
    pass


def data_generator(corpus):  # 训练数据生成器
    x, y = [], []
    for sentence in corpus:
        sentence = [0] * window + [word2id[w] for w in sentence if w in word2id] + [0] * window
        # 上面这句代码的意思是，因为我们是通过滑窗的方式来获取训练数据的，那么每一句语料的第一个词和最后一个词
        # 如何出现在中心位置呢？答案就是给它padding一下，例如“我/喜欢/足球”，两边分别补窗口大小个pad，得到“pad pad 我 喜欢 足球 pad pad”
        # 那么第一条训练数据的背景词就是['pad', 'pad','喜欢', '足球']，中心词就是'我'
        for i in range(window, len(sentence) - window):
            x.append(sentence[i - window: i] + sentence[i + 1: window + i + 1])
            y.append([sentence[i]] + get_negtive_sample(sentence[i], nb_word, nb_negative))
    x, y = np.array(x), np.array(y)
    z = np.zeros((len(x), nb_negative + 1))
    z[:, 0] = 1
    return x, y, z


def build_model(nb_word):
    # 苏神对多维向量或者叫张量的操作简直信手拈来，苏神经常使用这个K（keras.backend）对张量进行维度变换、维度提取和张量加减乘除。
    # 我这个小白看的是晕头转向，得琢磨半天。但是后来我也没有找到合适的方式来替换这个K，只能跟着用。
    # 第一个输入是周围词
    input_words = Input(shape=(window * 2,), dtype='int32')
    # 建立周围词的Embedding层
    input_vecs = Embedding(nb_word, word_size, name='word2vec')(input_words)
    # CBOW模型，直接将上下文词向量求和
    input_vecs_sum = Lambda(lambda x: K.sum(x, axis=1))(input_vecs)
    # 第二个输入，中心词以及负样本词
    samples = Input(shape=(nb_negative + 1,), dtype='int32')
    # 同样的，中心词和负样本词也有一个Emebdding层，其shape为 (?, nb_word, word_size)
    softmax_weights = Embedding(nb_word, word_size, name='W')(samples)
    softmax_biases = Embedding(nb_word, 1, name='b')(samples)
    # 将加和得到的词向量与中心词和负样本的词向量分别进行点乘
    # 注意到使用了K.expand_dims，这是为了将input_vecs_sum的向量推展一维，才能和softmax_weights进行dot
    input_vecs_sum_dot_ = Lambda(lambda x: K.batch_dot(x[0], K.expand_dims(x[1], 2)))([softmax_weights, input_vecs_sum])
    # 然后再将input_vecs_sum_dot_与softmax_biases进行相加，相当于 y = wx+b中的b项
    # 这里又用到了K.reshape，在将向量加和之后，得到的shape是(?, nb_negative+1, 1)，需要将其转换为(?, nb_negative+1)，才能进行softmax计算nb_negative+1个概率值
    add_biases = Lambda(lambda x: K.reshape(x[0] + x[1], shape=(-1, nb_negative + 1)))(
        [input_vecs_sum_dot_, softmax_biases])
    # 这里苏神用了K.softmax，而不是dense(nb_negative+1, activate='softmax')
    # 这是为什么呢？因为dense是先将上一层的张量先进行全联接，再使用softmax，而向下面这样使用K.softmax，就没有了全联接的过程。
    # 实验下来，小编尝试使用dense（activate='softmax')训练出来的效果很差。
    softmax = Lambda(lambda x: K.softmax(x))(add_biases)
    # 编译模型
    model = Model(inputs=[input_words, samples], outputs=softmax)
    # 使用categorical_crossentropy多分类交叉熵作损失函数
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.save_weights('word2vec.model')
    # embedding层的权重永远在模型的第一层
    embeddings = model.get_weights()[0]
    pass


def cbow():
    # TODO： cbow的输入应该是分好词的语料吗？

    pass


def most_similar(w):
    v = embeddings[word2id[w]]
    sims = np.dot(embeddings, v)
    sort = sims.argsort()[::-1]
    sort = sort[sort > 0]
    return [(id2word[i], sims[i]) for i in sort[:10]]


def main():
    # f
    pass


if __name__ == "__main__":
    main()
