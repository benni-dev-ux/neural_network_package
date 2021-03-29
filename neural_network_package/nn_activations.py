import numpy as np


def sigmoid(z):
    """Returns sigmoid of given z"""
    return 1 / (1 + np.e ** -z)


def sigmoid_deriv(z):
    """Returns sigmoid derivative of given z"""
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    """Returns tanh of given z"""
    return np.tanh(z)


def tanh_deriv(z):
    """Returns tanh derivative of given z"""
    return 1 - np.tanh(z) ** 2


def relu(z):
    """Returns relu of given z"""
    return np.maximum(0, z)


def relu_deriv(z):
    """Returns relu derivative of given z"""
    return (z > 0).astype(float)


ACTIVATION_FUNCTION = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu
}

ACTIVATION_FUNCTION_DERIV = {
    'sigmoid': sigmoid_deriv,
    'tanh': tanh_deriv,
    'relu': relu_deriv
}
