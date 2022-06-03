import numpy as np


def cross_entropy_error(y, t):
    if y.ndim == 1: # 如果是一维的, 比如说[0.1, 0.6, 0.3]
        y = y.reshape(1, y.size) # 转为二维的，[[0.1, 0.6, 0.3]]
        t = t.reshape(1, t.size)
    
    if t.size == y.size:
        t = t.argmax(axis=1)    # t现在成了标量？

    batch_size = y.shape[0]
    
    # loss_sum = 0
    # for bs in range(batch_size):
    #     cur_sum = 0 
    #     for k in range(y.shape[1]):
    #         cur_sum = t[k]*np.log(y[k])
    #     loss_sum += cur_sum
    # loss = - loss_sum / batch_size
    
    loss = - np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    # 这一串公式啥意思？和上面的实现等价吗？
    return loss

        