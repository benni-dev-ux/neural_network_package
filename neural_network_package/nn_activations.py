import numpy as np


def sigmoid(z):
    """Returns sigmoid of given z"""
    return 1 / (1 + np.e ** -z)


def tanh(z):
    """ returns tan of given z"""
    return np.tanh(z)


def relu(z):
    """Rectifying Linear Unit"""
    return np.maximum(0, z)
