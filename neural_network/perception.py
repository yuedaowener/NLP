import numpy as np

def AND(x1, x2):
    """与门"""
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    tmp = np.sum(x, w) + b
    return tmp > 0


def NAND(x1, x2):
    """与非门"""
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    tmp = np.sum(x, w) + b
    return tmp > 0


def OR(x1, x2):
    """或门"""
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    tmp = np.sum(x, w) + b
    return tmp > 0


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


def perception():
    pass


def main():
    pass



