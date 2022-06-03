import numpy as  np

def step_function(x:np.array):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # 会对各个元素进行计算


def softmax(x):
    c  = np.max(x) # 防止溢出

    if x.ndim == 2: # ndim是个啥?有这个属性吗?
        x = x - x.max(axis=1, keepdims=True) 
        x = np.exp(x)
        y = x / x.sum(axis=1, keepdims=True) # bug: 这个怎么防止溢出？

        # 按数据个数平均
        
    elif x.ndim == 1:
        exp_x = np.exp(x - c)
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x
    
    return y


def identity_function(x ):
    return x


def relu(x):
    return np.maxinum(0, x)
    pass


