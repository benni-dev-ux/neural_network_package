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
    # normalize Data, otherwise too big to compute e**inf -> inf
    o_norm = o - np.nanmax(o, axis=1).reshape(-1, 1)
    exp = np.e ** o_norm
    sum_all = np.sum(exp, axis=1)[:, np.newaxis]
    # clip, otherwise problems with .../0, but should not be necessary
    sum_all = np.clip(sum_all, a_min=0.000000001, a_max=None)
    return exp / sum_all

