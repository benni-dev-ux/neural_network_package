import numpy as np


def cross_entropy(h, y):
    """Cross Entropy Loss Function"""
    h = np.clip(h, a_min=0.000000001, a_max=None)
    return np.mean(- y * np.log(h) - (1 - y) * np.log(1 - h))


def categorical_cross_entropy(h, y_one_hot):
    """Cross Entropy Loss Function for Multiclass Neural Networks"""
    h = np.clip(h, a_min=0.000000001, a_max=None)
    return np.mean(-np.sum(y_one_hot * np.log(h), axis=1))


def softmax(o):
    """ Softmax converts arbitrary outputs into probabilities with a sum of 1"""
    return np.e ** o / np.sum(np.e ** o, axis=1)[:, np.newaxis]
