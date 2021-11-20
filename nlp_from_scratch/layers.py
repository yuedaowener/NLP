from functions import softmax, cross_entropy_error
import numpy as np


class MatMul:
    def __init__(self, W) -> None:
        self.params = [W]   # W直接就给了？不需要自己初始化？
        self.grads = [np.zeros_like(W)]
        self.x = None


    def forward(self, x):
        self.x = x
        w, = self.params
        y = np.dot(x, w)  
        return y
    
    def backward(self, dy): # dy是指网络最终输出L对y的导数
        """
        在当前层进行反向传播。输入是一个导数，输出还是一个导数。
        """
        w, = self.params
        dx = np.dot(dy, w.T) # dx表示网络最终输出L对x的导数。
        # y= wx可知，y对x的导数是w，再由链式法则知道y本身也是L的变量，所以求L对x的导数的时候，要乘以dy
        # 导数的行传和原来变量的形状是一样的 dx.shape = (nd), n表示有n个样本，d表示样本的大小。
        # 为了得到(n, d)形状的dx, 就得找到这样形状的矩阵相乘：(n, ?) * (?, d)
        # dy 的形状是(n, h), w的形状是(d, h), 将w转置。
        # 到这里其实就完了，可以返回，但是w也是变量，它的导数也应该保存在层本身的梯度里
        dw = np.dot(self.x.T, dy)
        # 形状 (nd) (dh) = (nh) 导数也一样。
        # dw = x * dy
        # 形状(d, h) = x.T dy
        self.grads[0][...] = dw  # 是深复制：将前者所指的地址里的值变成后者所指地址里的值。
        # 用self.gras[0] = dw 是浅复制：将前者也指向后者所指的地址。
        return dx
        
 
    

class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.params, self.grads = [], []
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t # forward要t干啥？
        self.y = softmax(x)

        if self.t.size == self.y.size:
            # 如果tag是one-hot, 转成类别
            self.t = self.t.argmax(axis=1)

        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dy=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1  # 为啥-1？
        dx *= dy # todo:softmax的导数
        dx = dx / batch_size
        return dx
        
        




        
        
