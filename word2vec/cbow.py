import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np

from neural_network.layers import MatMul, SoftmaxWithLoss
from neural_network.optimizer import Adam
from dataset import ptb
from dataset.hello import hello 

from preprocess import build_corpus, create_context_target, convert_one_hot
from neural_network.trainer import Trainer


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size) -> None:
        V, H = vocab_size, hidden_size
        self.V, self.H = V, H

        # 两层权重
        w_in = 0.01 * np.random.randn(V, H).astype('f')
        w_out = 0.01 * np.random.randn(H, V).astype('f')

        # 构建网络
        self.layer_in0 = MatMul(w_in)
        self.layer_in1 = MatMul(w_in)
        self.layer_out = MatMul(w_out)
        self.layer_loss = SoftmaxWithLoss()
        self.layers = [self.layer_in0, self.layer_in1, self.layer_out, self.layer_loss]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)
        
        self.word_vecs = w_in
        self.score = None
        self.loss = None
    

    def forward(self, contexts, targets):

        out0 = self.layer_in0.forward(contexts[:, 0]) # 前面那个词
        out1 = self.layer_in1.forward(contexts[:, 1])
        h = 0.5 * (out0 + out1)
        score = self.layer_out.forward(h)
        loss = self.layer_loss.forward(score, targets)
        self.score = score
        self.loss = loss
        return loss

    def backward(self, dloss=1):
        dscore = self.layer_loss.backward(dloss)
        dadd = self.layer_out.backward(dscore) * 0.5
        self.layer_in1.backward(dadd)
        self.layer_in0.backward(dadd)
        return None


def main():
    corpus, word_to_id, id_to_word = build_corpus(hello)
    # corpus, word_to_id, id_to_word = ptb.load_data('test')
    corpus_size = len(corpus)

    window_size = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    contexts, target = create_context_target(corpus, window_size)
    vocab_size = len(word_to_id)

    print("Converting to onehot ...")
    contexts_onehot = convert_one_hot(contexts, vocab_size)
    target_onehot = convert_one_hot(target, vocab_size) 
    # print(target)
    # print(contexts)
    # print(target_onehot)
    # print(contexts_onehot)
    
    model = SimpleCBOW(vocab_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    print("Training CBOW ...")
    trainer.fit(contexts_onehot, target_onehot, max_epoch = max_epoch, batch_size=batch_size) # bug: loss上升
    trainer.plot()

    pass

if __name__ == '__main__':
    main()