import numpy as np
from layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size) -> None:
        V, H = vocab_size, hidden_size
        self.V, self.H = V, H

        # 两层权重
        w_in = 0.01 * np.random.randn(V, H).astype('f') # todo: 为什么要乘以0.01?
        w_out = 0.01 * np.random.randn(H, V).astype('f')

        # 构建网络
        self.layer_in0 = MatMul(w_in)
        self.layer_in1 = MatMul(w_in) # todo：有多少个词就造多少个输入层？那批次又是什么意思呢？
        self.layer_out = MatMul(w_out)
        self.layer_loss = SoftmaxWithLoss()
        self.layers = [self.layer_in0, self.layer_in1, self.layer_out, self.layer_loss]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)
        
        self.score = None
        self.loss = None
    

    # 一个神经网络模型怎么实现反向传播？
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
        self.layer_in1.backward(dadd)   # 不需要返回什么？
        self.layer_in0.backward(dadd)
        return None
        
        pass

       
    
    

def cbow_predict():
    c0 = np.array([[1, 0, 0, 0, 0, 0, 0]]) # 列向量？
    c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

    # 两层权重
    w_in = np.random.randn(7, 3)
    w_out = np.random.randn(3, 7)

    # 构建网络
    layer_in0 = MatMul(w_in)
    layer_in1 = MatMul(w_in) # todo：有多少个词就造多少个输入层？那批次又是什么意思呢？
    layer_out = MatMul(w_out)

    # 正向传播
    h0 = layer_in0.forward(c0)
    h1 = layer_in0.forward(c1)
    h = 0.5 * (h0 + h1)
    y = layer_out.forward(h)

    print(y)
    pass


def test_class_CBOW():
    c0 = np.array([[1, 0, 0, 0, 0, 0, 0]]) # 列向量？
    c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])
    
    cbow = SimpleCBOW(c0.shape[1], 3)
    y = cbow.forward(c0, c1)
    print(y)
    pass





def main():
    # cbow_predict()
    test_class_CBOW()
    pass


if __name__ == "__main__":
    main()
