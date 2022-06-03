
import numpy as np
from perception import AND, NAND, OR, XOR
from activation import sigmoid, identity_function

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])

    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(x):
    nework = init_network()

    W1, W2, W3 = nework['W1'], nework['W2'], nework['W3']
    b1, b2, b3 = nework['b1'], nework['b2'], nework['b3']

    y1 = np.dot(x, W1) + b1
    h1 = sigmoid(y1)

    y2 = np.dot(h1, W2) + b2
    h2 = sigmoid(y2)

    y3 = np.dot(h2, W3) + b3
    y = identity_function(y3)

    return y


def main():
    x = np.array([1.0, 0.5])
    y = forward(x)
    print(y)


if __name__ == "__main__":
    main()


