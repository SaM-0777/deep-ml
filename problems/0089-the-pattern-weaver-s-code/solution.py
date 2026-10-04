import numpy as np


def softmax(values):
    values = np.asarray(values, dtype=float)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def pattern_weaver(n, crystal_values, dimension):
    X = np.asarray(crystal_values, dtype=float).reshape(n, dimension)
    scores = X @ X.T
    attention = softmax(scores)
    output = attention @ X

    return np.round(output.flatten(), 5)
