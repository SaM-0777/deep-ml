import numpy as np


def softmax(values):
    values = np.asarray(values, dtype=float)

    values = values - np.max(values, axis=-1, keepdims=True)

    exp_values = np.exp(values)

    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def pattern_weaver(n, crystal_values, dimension):
    X = np.asarray(crystal_values, dtype=float).reshape(n, dimension)

    # Similarity scores between every pair of crystals
    scores = X @ X.T

    # Attention weights
    attention = softmax(scores)

    # Weighted values
    output = attention @ X

    return np.round(output.flatten(), 5)