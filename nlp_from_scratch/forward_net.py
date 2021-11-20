import numpy as np

class Sigmoid:
    def __init__(self) -> None:
        self.params = []
    
    def forward(self, x):
        return 1 / (1 + np.exp(-x))     # todo: 这里的矩阵运算是怎么回事？


class Affine:
    def __init__(self, W, b) -> None:
        self.params = [W, b]
    
    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size) -> None:
        I, H, O = input_size, hidden_size, output_size

        # 初始化权重和偏置
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)     # bias是怎么加到矩阵值里面的？debug看一下。
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 层结构
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 将所有的参数放在一起
        self.params = [layer.params for layer in self.layers]
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x



def main():
    x = np.random.randn(10, 2)
    model = TwoLayerNet(2, 4, 3) # 数据的长度，中间层的个数，最后的分类数。
    s = model.predict(x)
    print(s)


if __name__ == "__main__":
    main()
        


