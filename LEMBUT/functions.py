import numpy as np


def linear(x):
    return x


def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocate
    # Zeros has to zero-out the array after allocation, no need for that
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def tanh(x):
    return np.tanh(x)


def dsigmoid(x):
    return x * (1 - x)


def dsoftmax(x):
    e_xi = np.exp(x)
    sum_ex = np.sum(x)
    return e_xi * (sum_ex - e_xi) / (sum_ex ** 2)


def drelu(x):
    return (x > 0) * 1.0


def dlinear(x):
    return 1


ACTIVATION_FUNCTIONS = {
    "linear": linear,
    "sigmoid": sigmoid,
    "relu": relu,
    "softmax": softmax,
    "tanh": tanh,
    "dsigmoid": dsigmoid,
    "drelu": drelu,
    "dlinear": dlinear,
    "dsoftmax": dsoftmax
}
